# detectors/pose.py
"""
detectors/pose.py
Estimation de l'orientation de la tête (pitch, yaw, roll) via Mediapipe
en utilisant cv2.solvePnP pour une meilleure précision.
"""
try:
    import mediapipe as mp
except ImportError:
    mp = None  # Mediapipe sera vérifié avant utilisation

import cv2
import numpy as np
import math

# Points 3D de référence du modèle de tête.
# Ces points sont génériques et ne correspondent pas à une personne spécifique.
# L'origine (0,0,0) est généralement au bout du nez.
model_points_3d = np.array([
    [0.0, 0.0, 0.0],  # Nose tip
    [0.0, -330.0, -65.0],  # Chin (distance approximative en unités génériques)
    [-225.0, 170.0, -135.0],  # Left eye left corner (MODEL's left)
    [225.0, 170.0, -135.0],  # Right eye right corner (MODEL's right)
    [-150.0, -150.0, -125.0],  # Left Mouth corner (MODEL's left)
    [150.0, -150.0, -125.0]  # Right mouth corner (MODEL's right)
], dtype=np.float64)

# Indices des landmarks MediaPipe correspondant aux model_points_3d.
# IMPORTANT: La "gauche" et la "droite" de MediaPipe sont souvent du point de vue du spectateur.
# Il faut s'assurer que ces indices correspondent correctement aux points du modèle 3D.
# 1. Bout du nez
# 2. Menton (point le plus bas)
# 3. Coin EXTERNE de l'œil GAUCHE du modèle (œil droit du spectateur)
# 4. Coin EXTERNE de l'œil DROIT du modèle (œil gauche du spectateur)
# 5. Coin GAUCHE de la bouche du modèle (coin droit de la bouche du spectateur)
# 6. Coin DROIT de la bouche du modèle (coin gauche de la bouche du spectateur)
#
# Indices MediaPipe Face Mesh (liste complète: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
#   Nose tip: 1
#   Chin: 152 (ou 175, 199) - point le plus bas du menton
#   Viewer's Left eye outer corner (MP Right): 33
#   Viewer's Right eye outer corner (MP Left): 263
#   Viewer's Left mouth corner (MP Right): 61
#   Viewer's Right mouth corner (MP Left): 291

# Mapping pour solvePnP:
# Model Point             | MediaPipe Landmark Index | Description
# ------------------------|--------------------------|----------------------------
# Nose tip                | 1                        | Nose tip
# Chin                    | 152                      | Chin bottom
# MODEL's Left eye corner | 33                       | Viewer's LEFT eye, outer corner (MP right eye landmarks group)
# MODEL's Right eye corner| 263                      | Viewer's RIGHT eye, outer corner (MP left eye landmarks group)
# MODEL's Left mouth corner| 61                      | Viewer's LEFT mouth, outer corner (MP right mouth landmarks group)
# MODEL's Right mouth corner| 291                     | Viewer's RIGHT mouth, outer corner (MP left mouth landmarks group)
solvepnp_landmark_indices = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(image_bgr: np.ndarray, face_box_xyxy: tuple):
    """
    Estime l'orientation de la tête (pitch, yaw, roll) à partir d'un rectangle facial.
    Utilise MediaPipe pour la détection des landmarks et cv2.solvePnP pour l'estimation de la pose.

    :param image_bgr: L'image complète en BGR (OpenCV format).
    :param face_box_xyxy: Un tuple (x1, y1, x2, y2) définissant le rectangle du visage
                           dans les coordonnées de l'image_bgr.
    :returns: Un tuple (pitch, yaw, roll) en degrés, ou None si l'estimation échoue.
    """
    if mp is None:
        # print("[WARN] detectors.pose: Mediapipe module not found. Pose estimation skipped.")
        return None  # Retourne None si mediapipe n'est pas là, plutôt que (0,0,0)

    x1, y1, x2, y2 = face_box_xyxy

    # Vérification de la validité du ROI
    if x1 >= x2 or y1 >= y2:
        # print(f"[DEBUG] detectors.pose: Invalid face_box ROI: {face_box_xyxy}. Skipping.")
        return None

    face_roi_bgr = image_bgr[y1:y2, x1:x2]

    if face_roi_bgr.size == 0:
        # print(f"[DEBUG] detectors.pose: Face ROI is empty for face_box: {face_box_xyxy}. Skipping.")
        return None

    # Obtenir les dimensions du ROI pour dénormaliser les landmarks plus tard
    roi_h, roi_w = face_roi_bgr.shape[:2]

    # Initialiser MediaPipe FaceMesh.
    # Utiliser 'with' pour une gestion correcte des ressources.
    # static_image_mode=True car nous traitons des images/frames individuelles.
    # max_num_faces=1 car nous avons déjà un ROI d'un seul visage.
    # refine_landmarks=True pour des coordonnées de landmarks plus précises.
    try:
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            # Convertir le ROI BGR en RGB pour MediaPipe
            face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
            face_roi_rgb.flags.writeable = False  # Optimisation: marquer comme non modifiable
            results = face_mesh.process(face_roi_rgb)
            face_roi_rgb.flags.writeable = True

            if not results.multi_face_landmarks:
                # print(f"[DEBUG] detectors.pose: No landmarks detected in ROI for face_box: {face_box_xyxy}")
                return None

            # Extraire les landmarks du premier (et unique) visage détecté dans le ROI
            # Les coordonnées (x,y,z) des landmarks sont normalisées par rapport aux dimensions du ROI.
            # x, y sont normalisés par rapport à la largeur et hauteur du ROI.
            # z est la profondeur, avec z=0 au centre approximatif de la tête.
            face_landmarks = results.multi_face_landmarks[0]

            # Collecter les points 2D pour solvePnP à partir des indices spécifiés
            image_points_2d = np.array([
                (face_landmarks.landmark[idx].x * roi_w, face_landmarks.landmark[idx].y * roi_h)
                for idx in solvepnp_landmark_indices
            ], dtype=np.float64)

            # Configuration de la caméra (intrinsèques)
            # Ces valeurs sont des approximations si la caméra n'est pas calibrée.
            focal_length_approx = float(roi_w)  # Approximation courante
            camera_center_x = roi_w / 2.0
            camera_center_y = roi_h / 2.0

            camera_matrix = np.array([
                [focal_length_approx, 0, camera_center_x],
                [0, focal_length_approx, camera_center_y],
                [0, 0, 1]
            ], dtype=np.float64)

            # Coefficients de distorsion (supposés nuls si non calibrés)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Résoudre la pose 3D avec solvePnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points_3d,
                image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE  # ou cv2.SOLVEPNP_SQPNP pour plus de rapidité
            )

            if not success:
                # print(f"[DEBUG] detectors.pose: cv2.solvePnP failed for face_box: {face_box_xyxy}")
                return None

            # Convertir le vecteur de rotation en matrice de rotation
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Calculer les angles d'Euler (pitch, yaw, roll) à partir de la matrice de rotation
            # sy = sqrt(R_mat[0,0] * R_mat[0,0] +  R_mat[1,0] * R_mat[1,0])
            # singular = sy < 1e-6
            # if not singular:
            #     x_pitch = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
            #     y_yaw   = math.atan2(-rotation_matrix[2,0], sy)
            #     z_roll  = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            # else:
            #     x_pitch = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            #     y_yaw   = math.atan2(-rotation_matrix[2,0], sy) # Can still use this
            #     z_roll  = 0

            # Alternative plus directe pour obtenir les angles d'Euler (attention aux conventions)
            # La fonction decomposeProjectionMatrix est plus robuste pour cela.
            # On a besoin d'une matrice de projection 3x4 P = K * [R | t]
            # K est la camera_matrix, R est rotation_matrix, t est translation_vector
            # P_manual = camera_matrix @ np.hstack((rotation_matrix, translation_vector)) # Pas K ici, juste R|t
            P_manual = np.hstack((rotation_matrix, translation_vector))

            # decomposeProjectionMatrix retourne: cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles
            # Les angles d'Euler sont en degrés.
            # Ordre: Pitch (autour de X), Yaw (autour de Y), Roll (autour de Z)
            try:
                eulerAngles = cv2.decomposeProjectionMatrix(P_manual)[6]
            except cv2.error:  # Peut arriver si la matrice n'est pas bien conditionnée
                # print(f"[DEBUG] detectors.pose: cv2.decomposeProjectionMatrix failed for face_box: {face_box_xyxy}")
                return None

            pitch = eulerAngles[0, 0]
            yaw = eulerAngles[1, 0]
            roll = eulerAngles[2, 0]

            # Ajustement optionnel des plages d'angles si nécessaire
            # Par exemple, le yaw de solvePnP peut parfois être dans une plage non intuitive.
            # Un ajustement commun pour le yaw pour le ramener à [-90, 90] ou [-180, 180]
            # if yaw > 180: yaw -=360 # exemple
            # Ou pour le pitch si la caméra est inversée etc.

            return pitch, yaw, roll

    except Exception as e:
        # print(f"[ERROR] detectors.pose: Unexpected error during MediaPipe processing or solvePnP for face_box {face_box_xyxy}: {e}")
        # import traceback
        # traceback.print_exc()
        return None


if __name__ == '__main__':
    # Exemple d'utilisation (nécessite une image et une face_box)
    # Assurez-vous que mediapipe est installé: pip install mediapipe
    print("Testing pose.py (requires an image and a face bounding box)")
    print(f"Mediapipe imported: {'Yes' if mp else 'No'}")

    # Créez une image factice pour le test si vous n'en avez pas
    # dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # cv2.rectangle(dummy_image, (200, 100), (400, 350), (0,255,0), 2) # Dessiner une fausse face box
    # dummy_face_box = (200, 100, 400, 350) # x1, y1, x2, y2

    # Remplacez par une vraie image et une vraie détection de visage pour un test significatif
    # try:
    #     image_test_path = "path_to_your_test_image_with_a_face.jpg"
    #     test_image = cv2.imread(image_test_path)
    #     if test_image is None:
    #         print(f"Failed to load test image: {image_test_path}")
    #     else:
    #         # Simulez une face_box (obtenue par un détecteur de visage comme YOLO)
    #         # Ces coordonnées doivent être adaptées à votre image de test
    #         test_face_box = (150, 80, 450, 400) # Exemple: x1, y1, x2, y2

    #         cv2.rectangle(test_image, (test_face_box[0], test_face_box[1]), (test_face_box[2], test_face_box[3]), (0,255,0), 2)

    #         angles = estimate_head_pose(test_image, test_face_box)

    #         if angles:
    #             pitch, yaw, roll = angles
    #             print(f"Estimated Pose -> Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f} degrees")

    #             # Afficher sur l'image (optionnel)
    #             cv2.putText(test_image, f"Pitch: {pitch:.2f}", (test_face_box[0], test_face_box[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    #             cv2.putText(test_image, f"Yaw:   {yaw:.2f}", (test_face_box[0], test_face_box[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    #             cv2.putText(test_image, f"Roll:  {roll:.2f}", (test_face_box[0], test_face_box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    #         else:
    #             print("Pose estimation failed.")

    #         cv2.imshow("Test Pose Estimation", test_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # except Exception as main_e:
    #     print(f"Error in __main__ test block: {main_e}")
    pass  # Enlever le bloc de test pour l'intégration
