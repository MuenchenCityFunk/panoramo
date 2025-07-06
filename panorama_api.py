#!/usr/bin/env python3
"""
Flask-API für Video-Upload und Panorama-Erstellung
Railway-Deployment Version - Alles in einem Server
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import threading
import time

app = Flask(__name__)
CORS(app)  # Erlaubt Cross-Origin Requests

# Konfiguration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'MP4', 'AVI', 'MOV', 'MKV', 'WEBM'}

# NEU: Device-ID Unterstützung
def get_device_output_folder(device_id=None):
    """Erstellt device-spezifischen Output-Ordner"""
    if device_id:
        device_folder = os.path.join(OUTPUT_FOLDER, device_id)
        os.makedirs(device_folder, exist_ok=True)
        return device_folder
    return OUTPUT_FOLDER

# Erstelle Ordner
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_gps_coordinates():
    """Holt automatisch die aktuellen GPS-Koordinaten"""
    try:
        import requests
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "lat": float(data.get('latitude', 52.52)),
                "lon": float(data.get('longitude', 13.405)),
                "accuracy": 1000.0,
                "timestamp": datetime.now().isoformat(),
                "source": "ip_geolocation"
            }
    except:
        pass
    
    return {
        "lat": 52.52,
        "lon": 13.405,
        "accuracy": 10000.0,
        "timestamp": datetime.now().isoformat(),
        "source": "fallback_default"
    }

def create_panorama_async(video_path, gps_coords, device_id=None):
    """Erstellt Panorama asynchron"""
    try:
        print(f"🚀  Starte Panorama-Erstellung für: {video_path}")
        print(f"🆔 Device ID: {device_id}")
        
        # NEU: Device-spezifischen Output-Ordner verwenden
        device_output_folder = get_device_output_folder(device_id)
        
        # Importiere Panorama-Funktionen
        from imgstitch.video_to_panorama import create_panorama_from_video_smart, get_video_info, create_depth_map_from_panorama
        
        # Video-Informationen
        video_info = get_video_info(video_path)
        
        # Panorama erstellen - NEU: Device-spezifischer Ordner
        panorama_result = create_panorama_from_video_smart(
            video_path=video_path,
            output_folder=device_output_folder,  # NEU: Device-Ordner
            create_depth_map=False,  # Wir erstellen es manuell
            confidence_threshold=0
        )
        
        # NEU: Prüfe ob panorama_result ein Dict ist (mit Statistiken) oder ein String (Pfad)
        if isinstance(panorama_result, dict):
            panorama_path = panorama_result["file_path"]
            stitching_stats = panorama_result
        else:
            panorama_path = panorama_result
            stitching_stats = None
        
        if panorama_path and os.path.exists(panorama_path):
            print(f"✅ Panorama erfolgreich erstellt: {panorama_path}")
            
            # DEPTH MAP ERSTELLEN - NEU: Device-spezifischer Ordner
            print(f"🗺️  Erstelle Depth Map für: {panorama_path}")
            depth_map_created = False
            depth_map_path = None
            
            try:
                depth_map_info = create_depth_map_from_panorama(panorama_path, device_output_folder)  # NEU: Device-Ordner
                
                # Robuste Prüfung des Rückgabewerts
                if depth_map_info is not None and isinstance(depth_map_info, dict):
                    depth_map_path = depth_map_info.get('depth_map_path', None)
                    if depth_map_path and os.path.exists(depth_map_path):
                        print(f"✅ Depth Map erstellt: {depth_map_path}")
                        depth_map_created = True
                    else:
                        print(f"⚠️  Depth Map Datei nicht gefunden: {depth_map_path}")
                else:
                    print(f"⚠️  Depth Map-Erstellung fehlgeschlagen: Rückgabewert ist None oder kein Dict")
                    
            except Exception as e:
                print(f"⚠️  Depth Map-Erstellung fehlgeschlagen: {e}")
                import traceback
                traceback.print_exc()
            
            # VIDEO LÖSCHEN (nur wenn Panorama erfolgreich)
            # Wir löschen das Video auch wenn Depth Map fehlschlägt, da Panorama wichtig ist
            try:
                os.remove(video_path)
                print(f"🗑️  Video gelöscht: {os.path.basename(video_path)}")
            except Exception as e:
                print(f"⚠️  Fehler beim Löschen des Videos: {e}")
            
            # Metadaten erstellen
            metadata = {
                "panorama_file": os.path.basename(panorama_path),
                "panorama_path": panorama_path,
                "gps_coordinates": gps_coords,
                "video_info": video_info,
                "created_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(panorama_path),
                "status": "completed",
                "device_type": "mobile_upload",
                "device_id": device_id,  # NEU: Device-ID hinzufügen
                "depth_map_created": depth_map_created,
                "depth_map_path": depth_map_path
            }
            
            # NEU: Stitching-Statistiken hinzufügen
            if stitching_stats:
                metadata["stitching_stats"] = {
                    "total_frames": stitching_stats.get("total_frames", 0),
                    "successful_frames": stitching_stats.get("successful_frames", 0),
                    "successful_stitches": stitching_stats.get("successful_stitches", 0),
                    "stitch_success_rate": stitching_stats.get("stitch_success_rate", 0)
                }
                print(f"📊 Stitching-Statistiken: {stitching_stats['successful_frames']}/{stitching_stats['total_frames']} Frames erfolgreich")
            
            # Bild-Informationen
            import cv2
            img = cv2.imread(panorama_path)
            if img is not None:
                metadata.update({
                    "image_width": img.shape[1],
                    "image_height": img.shape[0],
                    "image_channels": img.shape[2] if len(img.shape) > 2 else 1
                })
            
            # Zusätzliche Dateien (inkl. Depth Map) - NEU: Device-spezifischer Ordner
            additional_files = []
            for file in os.listdir(device_output_folder):  # NEU: Device-Ordner
                if file.startswith("depth_map_") or file.endswith(".gltf"):
                    file_path = os.path.join(device_output_folder, file)  # NEU: Device-Ordner
                    if os.path.exists(file_path):
                        additional_files.append({
                            "filename": file,
                            "file_size": os.path.getsize(file_path),
                            "type": "depth_map" if file.startswith("depth_map_") else "3d_model"
                        })
            
            metadata["additional_files"] = additional_files
            
            # Metadaten speichern - NEU: Device-spezifische Datei
            if device_id:
                metadata_file = os.path.join(device_output_folder, "panorama_metadata.json")  # NEU: Device-Ordner
            else:
                metadata_file = os.path.join(OUTPUT_FOLDER, "panorama_metadata.json")  # Fallback auf global
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Panorama erfolgreich erstellt: {panorama_path}")
            print(f" Depth Map erstellt: {depth_map_created}")
            print(f"🆔 Device ID: {device_id}")
            return True, metadata
        
        else:
            print(f"❌ Panorama-Erstellung fehlgeschlagen für: {video_path}")
            return False, None
            
    except Exception as e:
        print(f"❌ Fehler bei Panorama-Erstellung: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# NEUE ROUTEN für die Web-App
@app.route('/')
def index():
    """Hauptseite - serviert integrated-app.html"""
    return send_from_directory('.', 'integrated-app.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serviert alle statischen Dateien"""
    return send_from_directory('.', filename)

# API-ROUTEN mit /api Prefix
@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Empfängt Video-Upload und startet Panorama-Erstellung"""
    try:
        # Prüfe ob Video-Datei vorhanden
        if 'video' not in request.files:
            return jsonify({'error': 'Keine Video-Datei gefunden'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Keine Datei ausgewählt'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Nicht unterstütztes Video-Format'}), 400
        
        # GPS-Koordinaten aus Request oder automatisch holen
        gps_lat = request.form.get('gps_lat', None)
        gps_lon = request.form.get('gps_lon', None)
        
        # NEU: Device-ID aus Request holen
        device_id = request.form.get('device_id', None)
        print(f"🆔 Device ID: {device_id}")
        
        if gps_lat and gps_lon:
            gps_coords = {
                "lat": float(gps_lat),
                "lon": float(gps_lon),
                "accuracy": 10.0,
                "timestamp": datetime.now().isoformat(),
                "source": "mobile_gps"
            }
        else:
            gps_coords = get_current_gps_coordinates()
        
        # Video speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}_{file.filename}"
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        
        print(f"🎥 Video empfangen: {filename}")
        print(f"🌍 GPS-Koordinaten: {gps_coords['lat']}, {gps_coords['lon']}")
        
        # Panorama-Erstellung asynchron starten
        def process_panorama():
            success, metadata = create_panorama_async(video_path, gps_coords, device_id)  # NEU: device_id
            if success:
                print(f"✅ Panorama fertig: {metadata['panorama_file']}")
            else:
                print(f"❌ Panorama fehlgeschlagen für: {filename}")
        
        thread = threading.Thread(target=process_panorama)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Video erfolgreich hochgeladen. Panorama wird erstellt...',
            'filename': filename,
            'gps_coordinates': gps_coords,
            'device_id': device_id  # NEU: Device-ID zurückgeben
        })
        
    except Exception as e:
        print(f"❌ Upload-Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/panorama-status', methods=['GET'])
def panorama_status():
    """Gibt den Status der Panorama-Erstellung zurück"""
    try:
        metadata_file = os.path.join(OUTPUT_FOLDER, "panorama_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return jsonify(metadata)
        else:
            return jsonify({'status': 'no_panorama_found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/output/<filename>')
def serve_output(filename):
    """Serviert Dateien aus dem Output-Ordner"""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/output/<device_id>/<filename>')
def serve_device_output(device_id, filename):
    """Serviert device-spezifische Dateien aus dem Output-Ordner"""
    device_folder = os.path.join(OUTPUT_FOLDER, device_id)
    return send_from_directory(device_folder, filename)

if __name__ == '__main__':
    # Railway verwendet die PORT Umgebungsvariable
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Starte Server auf Port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 