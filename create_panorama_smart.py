#!/usr/bin/env python3
"""
Intelligentes Panorama-Erstellungsskript mit automatischer GPS-Erkennung
Erstellt automatisch Panoramen aus Videos mit aktuellen GPS-Daten
"""

import os
import sys
import json
import cv2
import socket
from datetime import datetime
from pathlib import Path

# Importiere deine Panorama-Funktionen
from imgstitch.video_to_panorama import create_panorama_from_video_smart, get_video_info

def check_api_server_running():
    """Prüft ob die Panorama-API bereits läuft"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5001))
        sock.close()
        return result == 0
    except:
        return False

def get_current_gps_coordinates():
    """Holt automatisch die aktuellen GPS-Koordinaten"""
    try:
        # Versuche GPS über verschiedene Methoden zu bekommen
        import requests
        
        # Methode 1: IP-basierte Geolokation
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "lat": float(data.get('latitude', 52.52)),
                    "lon": float(data.get('longitude', 13.405)),
                    "accuracy": 1000.0,  # IP-basierte Genauigkeit ist niedrig
                    "timestamp": datetime.now().isoformat(),
                    "source": "ip_geolocation"
                }
        except:
            pass
        
        # Methode 2: Fallback auf Standard-Koordinaten
        return {
            "lat": 52.52,  # Berlin als Fallback
            "lon": 13.405,
            "accuracy": 10000.0,
            "timestamp": datetime.now().isoformat(),
            "source": "fallback_default"
        }
        
    except Exception as e:
        print(f"⚠️  GPS-Erkennung fehlgeschlagen: {e}")
        # Fallback auf Standard-Koordinaten
        return {
            "lat": 52.52,
            "lon": 13.405,
            "accuracy": 10000.0,
            "timestamp": datetime.now().isoformat(),
            "source": "error_fallback"
        }

def save_panorama_metadata(panorama_path, gps_coords, video_info, output_folder):
    """Speichert Metadaten über das erstellte Panorama"""
    
    # Erstelle Metadaten
    metadata = {
        "panorama_file": os.path.basename(panorama_path),
        "panorama_path": panorama_path,
        "gps_coordinates": gps_coords,
        "video_info": video_info,
        "created_at": datetime.now().isoformat(),
        "file_size": os.path.getsize(panorama_path),
        "status": "completed",
        "device_type": "auto_processed"
    }
    
    # Lade Bild-Informationen
    try:
        img = cv2.imread(panorama_path)
        if img is not None:
            metadata.update({
                "image_width": img.shape[1],
                "image_height": img.shape[0],
                "image_channels": img.shape[2] if len(img.shape) > 2 else 1
            })
    except Exception as e:
        print(f"⚠️  Konnte Bild-Informationen nicht laden: {e}")
    
    # Prüfe auf zusätzliche Dateien (Depth Map, 3D-Modell)
    additional_files = []
    for file in os.listdir(output_folder):
        if file.startswith("depth_map_") or file.endswith(".gltf"):
            file_path = os.path.join(output_folder, file)
            additional_files.append({
                "filename": file,
                "file_size": os.path.getsize(file_path),
                "type": "depth_map" if file.startswith("depth_map_") else "3d_model"
            })
    
    metadata["additional_files"] = additional_files
    
    # Speichere Metadaten als JSON
    metadata_file = os.path.join(output_folder, "panorama_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Metadaten gespeichert: {metadata_file}")
    return metadata

def main():
    # Prüfe ob API-Server läuft
    if check_api_server_running():
        print("⚠️  Panorama-API läuft bereits auf Port 5001!")
        print("📱 Verwende die HTML-App zum Hochladen von Videos:")
        print("   1. Öffne integrated-app.html im Browser")
        print("   2. Klicke 'RECORD VIDEO' oder 'UPLOAD VIDEO'")
        print("   3. Das Panorama wird automatisch erstellt")
        print("\n💡 Das standalone Skript ist nur für manuelle Verarbeitung.")
        print("   Wenn du trotzdem fortfahren möchtest, stoppe die API zuerst.")
        return
    
    # Aktiviere Conda-Umgebung zuerst
    print("🚀 Intelligente Panorama-Erstellung gestartet")
    print("=" * 60)
    
    # Automatische GPS-Koordinaten holen
    print("📍 Hole aktuelle GPS-Koordinaten...")
    gps_coords = get_current_gps_coordinates()
    print(f"✅ GPS-Koordinaten: {gps_coords['lat']}, {gps_coords['lon']} (Quelle: {gps_coords['source']})")
    
    # Video-Datei finden - INTELLIGENTE LOGIK
    video_file = None
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM']
    
    # Prüfe zuerst den uploads-Ordner für neue Videos
    uploads_folder = "uploads"
    if os.path.exists(uploads_folder):
        print(f"🔍 Suche nach Videos im {uploads_folder}-Ordner...")
        
        # Sortiere Dateien nach Änderungsdatum (neueste zuerst)
        video_files = []
        for file in os.listdir(uploads_folder):
            if any(file.endswith(ext) for ext in video_extensions):
                file_path = os.path.join(uploads_folder, file)
                mod_time = os.path.getmtime(file_path)
                video_files.append((file_path, mod_time))
        
        if video_files:
            # Verwende das neueste Video
            video_files.sort(key=lambda x: x[1], reverse=True)
            video_file = video_files[0][0]
            print(f"🎥 Neuestes Video gefunden: {os.path.basename(video_file)}")
            print(f"📅 Erstellt: {datetime.fromtimestamp(video_files[0][1])}")
    
    # Fallback: Suche im aktuellen Verzeichnis (aber ignoriere panojuni.MP4)
    if not video_file:
        print("🔍 Suche nach Videos im aktuellen Verzeichnis...")
        for ext in video_extensions:
            for file in os.listdir('.'):
                if file.endswith(ext) and file != 'panojuni.MP4':  # Ignoriere altes Video
                    video_file = file
                    break
            if video_file:
                break
    
    if not video_file:
        print("❌ Keine Video-Datei gefunden!")
        print("   Stelle sicher, dass eine Video-Datei im uploads-Ordner oder aktuellen Ordner liegt.")
        print("   Unterstützte Formate:", ', '.join(video_extensions))
        print("\n💡 Tipp: Verwende die HTML-App für einfache Video-Uploads!")
        return
    
    print(f"🎥 Video gefunden: {video_file}")
    
    # Video-Informationen extrahieren
    try:
        video_info = get_video_info(video_file)
        print(f"📹 Video-Info: {video_info['duration']}s, {video_info['width']}x{video_info['height']}, {video_info['fps']}fps")
    except Exception as e:
        print(f"⚠️  Konnte Video-Informationen nicht laden: {e}")
        video_info = {"error": str(e)}
    
    # Output-Ordner erstellen
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        print("\n🏗️  Erstelle intelligentes Panorama...")
        print("   Dies kann einige Minuten dauern...")
        
        # Panorama erstellen
        panorama_path = create_panorama_from_video_smart(
            video_path=video_file,
            output_folder=output_folder,
            create_depth_map=True,
            confidence_threshold=0
        )
        
        if panorama_path and os.path.exists(panorama_path):
            print(f"\n✅ Panorama erfolgreich erstellt!")
            print(f"📁 Pfad: {panorama_path}")
            
            # DEPTH MAP ERSTELLEN
            print("\n🗺️  Erstelle Depth Map...")
            try:
                from imgstitch.video_to_panorama import create_depth_map_from_panorama
                depth_map_info = create_depth_map_from_panorama(panorama_path, output_folder)
                print(f"✅ Depth Map erstellt: {depth_map_info.get('depth_map_path', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Depth Map-Erstellung fehlgeschlagen: {e}")
            
            # Metadaten speichern
            metadata = save_panorama_metadata(panorama_path, gps_coords, video_info, output_folder)
            
            # Erfolgsmeldung
            print("\n" + "=" * 60)
            print("🎯 PANORAMA ERFOLGREICH ERSTELLT!")
            print("=" * 60)
            print(f"📁 Panorama: {panorama_path}")
            print(f"📍 GPS: {gps_coords['lat']}, {gps_coords['lon']}")
            print(f"📄 Metadaten: {os.path.join(output_folder, 'panorama_metadata.json')}")
            print("\n💡 Das Panorama kann jetzt automatisch geladen werden!")
            print("   Öffnen Sie integrated-app.html im Browser")
            print("=" * 60)
            
        else:
            print("❌ Panorama konnte nicht erstellt werden")
            
    except Exception as e:
        print(f"❌ Fehler bei der Panorama-Erstellung: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 