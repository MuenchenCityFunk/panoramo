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

def create_panorama_async(video_path, gps_coords):
    """Erstellt Panorama asynchron"""
    try:
        print(f"🚀  Starte Panorama-Erstellung für: {video_path}")
        
        # Importiere Panorama-Funktionen
        from imgstitch.video_to_panorama import create_panorama_from_video_smart, get_video_info, create_depth_map_from_panorama
        
        # Video-Informationen
        video_info = get_video_info(video_path)
        
        # Panorama erstellen
        panorama_path = create_panorama_from_video_smart(
            video_path=video_path,
            output_folder=OUTPUT_FOLDER,
            create_depth_map=False,  # Wir erstellen es manuell
            confidence_threshold=0
        )
        
        if panorama_path and os.path.exists(panorama_path):
            print(f"✅ Panorama erfolgreich erstellt: {panorama_path}")
            
            # DEPTH MAP ERSTELLEN
            print(f"🗺️  Erstelle Depth Map für: {panorama_path}")
            depth_map_created = False
            try:
                depth_map_info = create_depth_map_from_panorama(panorama_path, OUTPUT_FOLDER)
                print(f"✅ Depth Map erstellt: {depth_map_info.get('depth_map_path', 'N/A')}")
                depth_map_created = True
            except Exception as e:
                print(f"⚠️  Depth Map-Erstellung fehlgeschlagen: {e}")
                import traceback
                traceback.print_exc()
            
            # VIDEO LÖSCHEN (nur wenn Panorama UND Depth Map erfolgreich)
            if depth_map_created:
                try:
                    os.remove(video_path)
                    print(f"🗑️  Video gelöscht: {os.path.basename(video_path)}")
                except Exception as e:
                    print(f"⚠️  Fehler beim Löschen des Videos: {e}")
            else:
                print(f"📁 Video behalten (Depth Map fehlgeschlagen): {os.path.basename(video_path)}")
            
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
                "video_deleted": depth_map_created  # Track ob Video gelöscht wurde
            }
            
            # Bild-Informationen
            import cv2
            img = cv2.imread(panorama_path)
            if img is not None:
                metadata.update({
                    "image_width": img.shape[1],
                    "image_height": img.shape[0],
                    "image_channels": img.shape[2] if len(img.shape) > 2 else 1
                })
            
            # Zusätzliche Dateien (inkl. Depth Map)
            additional_files = []
            for file in os.listdir(OUTPUT_FOLDER):
                if file.startswith("depth_map_") or file.endswith(".gltf"):
                    file_path = os.path.join(OUTPUT_FOLDER, file)
                    additional_files.append({
                        "filename": file,
                        "file_size": os.path.getsize(file_path),
                        "type": "depth_map" if file.startswith("depth_map_") else "3d_model"
                    })
            
            metadata["additional_files"] = additional_files
            
            # Metadaten speichern
            metadata_file = os.path.join(OUTPUT_FOLDER, "panorama_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Panorama erfolgreich erstellt: {panorama_path}")
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
            success, metadata = create_panorama_async(video_path, gps_coords)
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
            'gps_coordinates': gps_coords
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

if __name__ == '__main__':
    # Railway verwendet die PORT Umgebungsvariable
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Starte Server auf Port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 