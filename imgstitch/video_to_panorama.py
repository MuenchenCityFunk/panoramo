import os
# Fix für OpenMP Konflikt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import subprocess
import tempfile
import shutil
from typing import List, Optional, Tuple
from .stitch_images import stitch_images_and_save  # Direkter Import der Funktion
from . import exceptions


def check_ffmpeg_installation() -> bool:
    """
    Prüft ob FFmpeg auf dem System installiert ist.
    
    Returns:
        bool: True wenn FFmpeg verfügbar ist, False sonst
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_frames_from_video(video_path: str, output_folder: str, frame_rate: Optional[float] = None, 
                            max_frames: Optional[int] = None) -> List[str]:
    """
    Extrahiert Frames aus einem Video mit FFmpeg.
    
    Args:
        video_path (str): Pfad zur Videodatei
        output_folder (str): Ordner für die extrahierten Frames
        frame_rate (float, optional): FPS für Frame-Extraktion (None = Original-FPS)
        max_frames (int, optional): Maximale Anzahl Frames zu extrahieren
    
    Returns:
        List[str]: Liste der extrahierten Frame-Dateinamen
    """
    # FFmpeg-Installation prüfen
    if not check_ffmpeg_installation():
        raise RuntimeError("FFmpeg ist nicht installiert. Bitte installieren Sie FFmpeg: https://ffmpeg.org/download.html")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Videodatei nicht gefunden: {video_path}")
    
    # Erstelle Output-Ordner falls nicht vorhanden
    os.makedirs(output_folder, exist_ok=True)
    
    # FFmpeg-Befehl zusammenbauen mit verbesserten Parametern
    filter_complex = "fps=1"  # Standard: 1 Frame pro Sekunde
    
    # Frame-Rate anpassen falls angegeben
    if frame_rate:
        filter_complex = f"fps={frame_rate}"
    
    # Maximale Frames begrenzen falls angegeben
    if max_frames:
        filter_complex = f"{filter_complex},select=lt(n\\,{max_frames})"
    
    # Zusätzliche Filter für bessere Qualität
    filter_complex = f"{filter_complex},scale=1280:720:flags=lanczos"
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', filter_complex,
        '-frame_pts', '1',
        '-q:v', '2',  # Hohe Qualität
        '-y',  # Überschreibe existierende Dateien
        os.path.join(output_folder, 'frame_%04d.jpg')
    ]
    
    try:
        # FFmpeg ausführen
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extrahiere Dateinamen der erstellten Frames
        frame_files = sorted([f for f in os.listdir(output_folder) 
                            if f.startswith('frame_') and f.endswith('.jpg')])
        
        if not frame_files:
            raise exceptions.InsufficientImagesError(0)
        
        # Überprüfe ob Frames gültig sind
        valid_frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(output_folder, frame_file)
            if os.path.getsize(frame_path) > 0:
                valid_frames.append(frame_file)
        
        if len(valid_frames) < 2:
            raise exceptions.InsufficientImagesError(len(valid_frames))
        
        return valid_frames
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg-Fehler: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Extrahieren der Frames: {str(e)}")


def create_panorama_from_video(video_path: str, stitch_direction: int = 1, 
                             frame_rate: Optional[float] = None, max_frames: Optional[int] = None,
                             output_folder: str = "output", temp_folder: Optional[str] = None) -> str:
    """
    Erstellt ein Panorama aus einem Video durch Frame-Extraktion und Stitching.
    
    Args:
        video_path (str): Pfad zur Videodatei
        stitch_direction (int): 1 für horizontales Stitching, 0 für vertikales Stitching
        frame_rate (float, optional): FPS für Frame-Extraktion
        max_frames (int, optional): Maximale Anzahl Frames zu extrahieren
        output_folder (str): Ordner für das finale Panorama
        temp_folder (str, optional): Temporärer Ordner für Frames (wird automatisch erstellt falls None)
    
    Returns:
        str: Pfad zum erstellten Panorama
    """
    # Temporären Ordner erstellen falls nicht angegeben
    if temp_folder is None:
        temp_folder = tempfile.mkdtemp(prefix="video_frames_")
        cleanup_temp = True
    else:
        os.makedirs(temp_folder, exist_ok=True)
        cleanup_temp = False
    
    try:
        # Frames extrahieren
        print(f"Extrahiere Frames aus Video: {video_path}")
        frame_files = extract_frames_from_video(video_path, temp_folder, frame_rate, max_frames)
        print(f"Extrahierte {len(frame_files)} Frames")
        
        # Panorama erstellen
        print("Erstelle Panorama aus Frames...")
        stitch_images_and_save(
            image_folder=temp_folder,
            image_filenames=frame_files,
            stitch_direction=stitch_direction,
            output_folder=output_folder
        )
        
        # Pfad zum erstellten Panorama finden
        panorama_files = [f for f in os.listdir(output_folder) 
                         if f.startswith('stitched_image_') and f.endswith('.jpg')]
        if panorama_files:
            panorama_path = os.path.join(output_folder, sorted(panorama_files)[-1])
            print(f"Panorama erfolgreich erstellt: {panorama_path}")
            return panorama_path
        else:
            raise RuntimeError("Panorama konnte nicht erstellt werden")
            
    finally:
        # Temporären Ordner aufräumen
        if cleanup_temp and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)


def create_panorama_from_video_advanced(video_path: str, stitch_direction: int = 1,
                                      frame_interval: int = 30, max_frames: Optional[int] = None,
                                      output_folder: str = "output", temp_folder: Optional[str] = None) -> str:
    """
    Erweiterte Version mit mehr Kontrolle über Frame-Extraktion.
    
    Args:
        video_path (str): Pfad zur Videodatei
        stitch_direction (int): 1 für horizontales Stitching, 0 für vertikales Stitching
        frame_interval (int): Extrahiere jeden n-ten Frame (z.B. 30 = jeden 30. Frame)
        max_frames (int, optional): Maximale Anzahl Frames zu extrahieren
        output_folder (str): Ordner für das finale Panorama
        temp_folder (str, optional): Temporärer Ordner für Frames
    
    Returns:
        str: Pfad zum erstellten Panorama
    """
    # FFmpeg-Installation prüfen
    if not check_ffmpeg_installation():
        raise RuntimeError("FFmpeg ist nicht installiert. Bitte installieren Sie FFmpeg: https://ffmpeg.org/download.html")
    
    if temp_folder is None:
        temp_folder = tempfile.mkdtemp(prefix="video_frames_")
        cleanup_temp = True
    else:
        os.makedirs(temp_folder, exist_ok=True)
        cleanup_temp = False
    
    try:
        # FFmpeg-Befehl für Frame-Intervall
        filter_complex = f"select=not(mod(n\\,{frame_interval}))"
        
        # Maximale Frames begrenzen falls angegeben
        if max_frames:
            filter_complex = f"{filter_complex},select=lt(n\\,{max_frames})"
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', filter_complex,
            '-vsync', 'vfr',
            '-y',  # Überschreibe existierende Dateien
            os.path.join(temp_folder, 'frame_%04d.jpg')
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            frame_files = sorted([f for f in os.listdir(temp_folder) 
                                if f.startswith('frame_') and f.endswith('.jpg')])
            
            if len(frame_files) < 2:
                raise exceptions.InsufficientImagesError(len(frame_files))
            
            print(f"Extrahiert {len(frame_files)} Frames (jeder {frame_interval}. Frame)")
            
            # Panorama erstellen
            stitch_images_and_save(
                image_folder=temp_folder,
                image_filenames=frame_files,
                stitch_direction=stitch_direction,
                output_folder=output_folder
            )
            
            # Pfad zum erstellten Panorama finden
            panorama_files = [f for f in os.listdir(output_folder) 
                             if f.startswith('stitched_image_') and f.endswith('.jpg')]
            if panorama_files:
                panorama_path = os.path.join(output_folder, sorted(panorama_files)[-1])
                print(f"Panorama erfolgreich erstellt: {panorama_path}")
                return panorama_path
            else:
                raise RuntimeError("Panorama konnte nicht erstellt werden")
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg-Fehler: {e.stderr}")
            
    finally:
        if cleanup_temp and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)


def validate_frames(frame_files: List[str], temp_folder: str) -> List[str]:
    """
    Überprüft, ob alle Frame-Dateien gültig sind und nicht leer sind.
    
    Args:
        frame_files (List[str]): Liste der Frame-Dateinamen
        temp_folder (str): Ordner mit den Frames
    
    Returns:
        List[str]: Liste der gültigen Frame-Dateinamen
    """
    valid_frames = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(temp_folder, frame_file)
        
        # Prüfe ob Datei existiert und nicht leer ist
        if not os.path.exists(frame_path):
            print(f"⚠️  Warnung: Frame-Datei nicht gefunden: {frame_file}")
            continue
            
        file_size = os.path.getsize(frame_path)
        if file_size == 0:
            print(f"⚠️  Warnung: Leere Frame-Datei: {frame_file}")
            continue
        
        # Versuche das Bild zu laden
        try:
            img = cv2.imread(frame_path)
            if img is None or img.size == 0:
                print(f"⚠️  Warnung: Ungültige Frame-Datei: {frame_file}")
                continue
            valid_frames.append(frame_file)
        except Exception as e:
            print(f"⚠️  Warnung: Fehler beim Laden von {frame_file}: {e}")
            continue
    
    if len(valid_frames) < 2:
        raise RuntimeError(f"Zu wenige gültige Frames gefunden: {len(valid_frames)} (mindestens 2 benötigt)")
    
    print(f"✅ {len(valid_frames)} von {len(frame_files)} Frames sind gültig")
    return valid_frames


def create_panorama_from_video_low_confidence(video_path: str, stitch_direction: int = 1, 
                                            frame_rate: Optional[float] = None, max_frames: Optional[int] = None,
                                            output_folder: str = "output", temp_folder: Optional[str] = None,
                                            confidence_threshold: int = 30) -> str:
    """
    Version mit reduziertem Confidence-Threshold für Videos mit wenig Überlappung.
    
    Args:
        video_path (str): Pfad zur Videodatei
        stitch_direction (int): 1 für horizontales Stitching, 0 für vertikales Stitching
        frame_rate (float, optional): FPS für Frame-Extraktion
        max_frames (int, optional): Maximale Anzahl Frames zu extrahieren
        output_folder (str): Ordner für das finale Panorama
        temp_folder (str, optional): Temporärer Ordner für Frames
        confidence_threshold (int): Reduzierter Confidence-Threshold (Standard: 30 statt 65)
    
    Returns:
        str: Pfad zum erstellten Panorama
    """
    # Temporären Ordner erstellen
    if temp_folder is None:
        temp_folder = tempfile.mkdtemp(prefix="video_frames_")
        cleanup_temp = True
    else:
        os.makedirs(temp_folder, exist_ok=True)
        cleanup_temp = False
    
    try:
        # Frames extrahieren
        print(f"Extrahiere Frames aus Video: {video_path}")
        frame_files = extract_frames_from_video(video_path, temp_folder, frame_rate, max_frames)
        print(f"Extrahierte {len(frame_files)} Frames")
        
        # Frames validieren
        print("Validiere Frames...")
        valid_frame_files = validate_frames(frame_files, temp_folder)
        
        # Temporär Confidence-Threshold reduzieren
        from . import utils
        original_threshold = utils.CONFIDENCE_THRESH
        utils.CONFIDENCE_THRESH = confidence_threshold
        
        try:
            # Panorama erstellen
            print("Erstelle Panorama aus Frames...")
            stitch_images_and_save(
                image_folder=temp_folder,
                image_filenames=valid_frame_files,  # Verwende nur gültige Frames
                stitch_direction=stitch_direction,
                output_folder=output_folder
            )
        finally:
            # Threshold zurücksetzen
            utils.CONFIDENCE_THRESH = original_threshold
        
        # Pfad zum erstellten Panorama finden
        panorama_files = [f for f in os.listdir(output_folder) 
                         if f.startswith('stitched_image_') and f.endswith('.jpg')]
        if panorama_files:
            panorama_path = os.path.join(output_folder, sorted(panorama_files)[-1])
            print(f"Panorama erfolgreich erstellt: {panorama_path}")
            return panorama_path
        else:
            raise RuntimeError("Panorama konnte nicht erstellt werden")
            
    finally:
        # Temporären Ordner aufräumen
        if cleanup_temp and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)


def get_video_info(video_path: str) -> dict:
    """
    Extrahiert Informationen über ein Video mit FFmpeg.
    
    Args:
        video_path (str): Pfad zur Videodatei
    
    Returns:
        dict: Dictionary mit Video-Informationen (duration, fps, resolution, etc.)
    """
    if not check_ffmpeg_installation():
        raise RuntimeError("FFmpeg ist nicht installiert")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Videodatei nicht gefunden: {video_path}")
    
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        
        # Extrahiere relevante Informationen
        video_info = {}
        
        # Format-Informationen
        if 'format' in info:
            format_info = info['format']
            video_info['duration'] = float(format_info.get('duration', 0))
            video_info['bitrate'] = format_info.get('bit_rate')
        
        # Video-Stream-Informationen
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_info['width'] = int(stream.get('width', 0))
                video_info['height'] = int(stream.get('height', 0))
                video_info['fps'] = eval(stream.get('r_frame_rate', '0/1'))  # z.B. "30/1" -> 30.0
                video_info['codec'] = stream.get('codec_name')
                break
        
        return video_info
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe-Fehler: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Lesen der Video-Informationen: {str(e)}")

def create_depth_map_from_panorama(panorama_path: str, output_folder: str = "output") -> dict:
    """
    Erstellt eine Depth Map und optional ein 3D-Modell aus einem Panorama.
    Funktioniert auch ohne Open3D (nur Depth Map).
    
    Args:
        panorama_path (str): Pfad zum Panorama
        output_folder (str): Ordner für die Ausgabe
    
    Returns:
        dict: Pfade zu den erstellten Dateien
    """
    try:
        print(f"Erstelle Depth Map für: {panorama_path}")
        
        # Import der benötigten Bibliotheken
        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        import torch
        import numpy as np
        from PIL import Image
        from pathlib import Path
        import os
        
        # Lade das DPT Modell (nur beim ersten Aufruf)
        if not hasattr(create_depth_map_from_panorama, 'feature_extractor'):
            print("Lade DPT Modell...")
            create_depth_map_from_panorama.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
            create_depth_map_from_panorama.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        
        # Lade das Panorama
        image_path = Path(panorama_path)
        image_raw = Image.open(image_path)
        
        # Resize für bessere Performance (optional)
        max_size = 1200  # Maximale Größe für bessere Performance
        if max(image_raw.size) > max_size:
            ratio = max_size / max(image_raw.size)
            new_size = (int(image_raw.size[0] * ratio), int(image_raw.size[1] * ratio))
            image = image_raw.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Bild auf {new_size} resized für bessere Performance")
        else:
            image = image_raw
        
        # Bereite das Bild für das Modell vor
        print("🔍 Analysiere Depth...")
        encoding = create_depth_map_from_panorama.feature_extractor(image, return_tensors="pt")
        
        # Forward Pass
        with torch.no_grad():
            outputs = create_depth_map_from_panorama.model(**encoding)
            predicted_depth = outputs.predicted_depth
        
        # Interpoliere auf Originalgröße
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = prediction.cpu().numpy()
        depth_image = (output * 255 / np.max(output)).astype("uint8")
        
        # Speichere Depth Map
        depth_map_path = os.path.join(output_folder, f"depth_map_{image_path.stem}.png")
        depth_img = Image.fromarray(depth_image)
        depth_img.save(depth_map_path)
        print(f"💾 Depth Map gespeichert: {depth_map_path}")
        
        # Erstelle 3D-Modell nur wenn Open3D verfügbar ist
        gltf_path = None
        try:
            import open3d as o3d
            print("🏗️  Erstelle 3D-Modell...")
            gltf_path = create_3d_obj_from_depth(
                np.array(image), depth_image, image_path, output_folder
            )
            print(f"💾 3D-Modell gespeichert: {gltf_path}")
        except ImportError:
            print("⚠️  Open3D nicht verfügbar - überspringe 3D-Modell-Erstellung")
            print("ℹ️  Depth Map wurde trotzdem erfolgreich erstellt")
        except Exception as e:
            print(f"⚠️  3D-Modell fehlgeschlagen, versuche mit reduzierter Qualität: {e}")
            try:
                gltf_path = create_3d_obj_from_depth(
                    np.array(image), depth_image, image_path, output_folder, depth=6
                )
                print(f"💾 3D-Modell (reduzierte Qualität) gespeichert: {gltf_path}")
            except Exception as e2:
                print(f"❌ 3D-Modell konnte nicht erstellt werden: {e2}")
                gltf_path = None
        
        return {
            'panorama': panorama_path,
            'depth_map': depth_map_path,
            '3d_model': gltf_path
        }
        
    except Exception as e:
        print(f"❌ Fehler bei der Depth Map Erstellung: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_3d_obj_from_depth(rgb_image, depth_image, image_path, output_folder, depth=10):
    """
    Erstellt ein 3D-Modell aus RGB- und Depth-Image.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("❌ Open3D nicht verfügbar - 3D-Modell-Erstellung nicht möglich")
        return None
    
    import numpy as np
    import os
    
    try:
        depth_o3d = o3d.geometry.Image(depth_image)
        image_o3d = o3d.geometry.Image(rgb_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            image_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        w = int(depth_image.shape[1])
        h = int(depth_image.shape[0])

        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(w, h, 500, 500, w / 2, h / 2)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

        print("   📐 Berechne Normalen...")
        pcd.normals = o3d.utility.Vector3dVector(
            np.zeros((1, 3))
        )  # invalidate existing normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 1000.0])
        )
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        print("   🏗️  Poisson Surface Reconstruction...")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=0, scale=1.1, linear_fit=True
            )

        voxel_size = max(mesh_raw.get_max_bound() - mesh_raw.get_min_bound()) / 256
        print(f"    Voxel-Größe = {voxel_size:e}")
        mesh = mesh_raw.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )

        bbox = pcd.get_axis_aligned_bounding_box()
        mesh_crop = mesh.crop(bbox)
        
        # Speichere im Output-Ordner
        gltf_filename = f"3d_model_{image_path.stem}.gltf"
        gltf_path = os.path.join(output_folder, gltf_filename)
        o3d.io.write_triangle_mesh(gltf_path, mesh_crop, write_triangle_uvs=True)
        
        return gltf_path
        
    except Exception as e:
        print(f"❌ Fehler bei 3D-Modell-Erstellung: {e}")
        return None


def create_panorama_from_video_smart(video_path: str, stitch_direction: int = 1, 
                                   frame_rate: Optional[float] = None, max_frames: Optional[int] = None,
                                   output_folder: str = "output", temp_folder: Optional[str] = None,
                                   confidence_threshold: int = 0, create_depth_map: bool = True) -> str:
    """
    Intelligente Version, die alle erfolgreichen Panoramen behält.
    Erstellt mehrere Panoramen und behält alle, die erfolgreich erstellt wurden.
    
    Args:
        video_path (str): Pfad zur Videodatei
        stitch_direction (int): 1 für horizontales Stitching, 0 für vertikales Stitching
        frame_rate (float, optional): FPS für Frame-Extraktion
        max_frames (int, optional): Maximale Anzahl Frames zu extrahieren
        output_folder (str): Ordner für das finale Panorama
        temp_folder (str, optional): Temporärer Ordner für Frames
        confidence_threshold (int): Confidence-Threshold (Standard: 0 für keine Prüfung)
        create_depth_map (bool): Erstellt automatisch eine Depth Map (Standard: True)
    
    Returns:
        str: Pfad zum neuesten erstellten Panorama
    """
    # Temporären Ordner erstellen
    if temp_folder is None:
        temp_folder = tempfile.mkdtemp(prefix="video_frames_")
        cleanup_temp = True
    else:
        os.makedirs(temp_folder, exist_ok=True)
        cleanup_temp = False
    
    try:
        # Frames extrahieren
        print(f"Extrahiere Frames aus Video: {video_path}")
        frame_files = extract_frames_from_video(video_path, temp_folder, frame_rate, max_frames)
        print(f"Extrahierte {len(frame_files)} Frames")
        
        # Frames validieren
        print("Validiere Frames...")
        valid_frame_files = validate_frames(frame_files, temp_folder)
        
        # Intelligentes Stitching
        print("Erstelle Panorama mit intelligentem Stitching...")
        
        # Temporär Confidence-Threshold reduzieren
        from . import utils
        original_threshold = utils.CONFIDENCE_THRESH
        utils.CONFIDENCE_THRESH = confidence_threshold
        
        created_panoramas = []
        
        try:
            # Erstelle mehrere Panoramen und behalte alle
            panorama_attempts = []
            
            # Panorama 1: Alle Frames
            try:
                print("Versuch 1: Alle Frames stitchen...")
                stitch_images_and_save(
                    image_folder=temp_folder,
                    image_filenames=valid_frame_files,
                    stitch_direction=stitch_direction,
                    output_folder=output_folder
                )
                panorama_attempts.append(("full", len(valid_frame_files)))
                print("✅ Vollständiges Panorama erstellt")
            except Exception as e:
                print(f"❌ Vollständiges Panorama fehlgeschlagen: {e}")
            
            # Panorama 2: Nur erste Hälfte der Frames
            if len(valid_frame_files) > 10:
                try:
                    print("Versuch 2: Erste Hälfte der Frames stitchen...")
                    half_frames = valid_frame_files[:len(valid_frame_files)//2]
                    stitch_images_and_save(
                        image_folder=temp_folder,
                        image_filenames=half_frames,
                        stitch_direction=stitch_direction,
                        output_folder=output_folder
                    )
                    panorama_attempts.append(("half", len(half_frames)))
                    print("✅ Halb-Panorama erstellt")
                except Exception as e:
                    print(f"❌ Halb-Panorama fehlgeschlagen: {e}")
            
            # Panorama 3: Nur erste 10 Frames
            if len(valid_frame_files) > 10:
                try:
                    print("Versuch 3: Erste 10 Frames stitchen...")
                    first_10 = valid_frame_files[:10]
                    stitch_images_and_save(
                        image_folder=temp_folder,
                        image_filenames=first_10,
                        stitch_direction=stitch_direction,
                        output_folder=output_folder
                    )
                    panorama_attempts.append(("first10", len(first_10)))
                    print("✅ Erste-10-Panorama erstellt")
                except Exception as e:
                    print(f"❌ Erste-10-Panorama fehlgeschlagen: {e}")
            
            # Panorama 4: Nur erste 5 Frames
            if len(valid_frame_files) > 5:
                try:
                    print("Versuch 4: Erste 5 Frames stitchen...")
                    first_5 = valid_frame_files[:5]
                    stitch_images_and_save(
                        image_folder=temp_folder,
                        image_filenames=first_5,
                        stitch_direction=stitch_direction,
                        output_folder=output_folder
                    )
                    panorama_attempts.append(("first5", len(first_5)))
                    print("✅ Erste-5-Panorama erstellt")
                except Exception as e:
                    print(f"❌ Erste-5-Panorama fehlgeschlagen: {e}")
            
        finally:
            # Threshold zurücksetzen
            utils.CONFIDENCE_THRESH = original_threshold
        
        # Finde alle erstellten Panoramen
        panorama_files = [f for f in os.listdir(output_folder) 
                         if f.startswith('stitched_image_') and f.endswith('.jpg')]
        
        if panorama_files:
            # Sortiere nach Erstellungszeit (neueste zuerst)
            panorama_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
            
            # Behalte alle Panoramen - lösche nichts!
            print(f" {len(panorama_files)} Panoramen erstellt und behalten:")
            for panorama_file in panorama_files:
                panorama_path = os.path.join(output_folder, panorama_file)
                try:
                    img = cv2.imread(panorama_path)
                    if img is not None:
                        print(f"   📊 {panorama_file}: Breite={img.shape[1]}, Höhe={img.shape[0]}")
                        created_panoramas.append(panorama_path)
                except Exception as e:
                    print(f"⚠️  Fehler beim Laden von {panorama_file}: {e}")
            
            # Gib das neueste Panorama zurück
            if created_panoramas:
                newest_panorama = created_panoramas[0]
                print(f" Neuestes Panorama: {os.path.basename(newest_panorama)}")
                return newest_panorama
            else:
                raise RuntimeError("Kein gültiges Panorama gefunden")
        else:
            raise RuntimeError("Kein Panorama konnte erstellt werden")
            
    finally:
        # Temporären Ordner aufräumen
        if cleanup_temp and os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
