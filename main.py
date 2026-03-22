import cv2
import os
import kagglehub

database = kagglehub.dataset_download("ruizgara/socofing")

sample_path = "test.BMP"
sample = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

if sample is None:
    print(f"Error: There is no sample file at: {sample_path}")
    exit()

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(sample, None)

database_path = os.path.join(database, 'SOCOFing', 'Real')

if not os.path.exists(database_path):
    print(f"Error: Folder '{database_path}' does not exist!")
    exit()

for file in os.listdir(database_path)[:100]:
    current_path = os.path.join(database_path, file)
    fingerprint_image = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)

    if fingerprint_image is None:
        continue

    keypoints2, descriptors2 = sift.detectAndCompute(fingerprint_image, None)

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher({'algorithm': FLANN_INDEX_KDTREE, 'trees': 10}, {})

    if descriptors1 is not None and descriptors2 is not None:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        match_points = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                match_points.append(m)

    num_keypoints = min(len(keypoints1), len(keypoints2))

    if num_keypoints > 0:
            score = len(match_points) / num_keypoints * 100
            
            if score > best_score:
                best_score = score
                filename = file
                matched_image = fingerprint_image
                kp1, kp2, mp = keypoints1, keypoints2, match_points

if filename:
    print(f"Mached image: {filename}")
    print(f"Similarity Score: {round(best_score, 2)}%")

    # Wizualizacja
    result = cv2.drawMatches(sample, kp1, matched_image, kp2, mp, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Images mached", cv2.resize(result, None, fx=1.5, fy=1.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No maches found.")