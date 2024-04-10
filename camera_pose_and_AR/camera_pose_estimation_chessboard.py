import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'my_chessboard_02.avi'
K =  np.array([[1.01212241e+03, 0.00000000e+00, 9.54081522e+02],
 [0.00000000e+00, 1.01205120e+03, 5.41382436e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([0.00020659,  0.02590867, -0.00105908,  0.00307626, -0.02384998])

board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

new_width = 640
new_height = 480

# Prepare a 3D box for simple AR
#box_lower = board_cellsize * np.array([[4, 2,  0], [5, 2,  0], [5, 4,  0], [4, 4,  0]])
#box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Define 3D points for a sphere
radius = 0.1  # 반지름
num_phi = 30  # 세로 방향 분할 수
num_theta = 60  # 가로 방향 분할 수

sphere_points = []
for i in range(num_phi + 1):
    phi = np.pi * i / num_phi
    for j in range(num_theta):
        theta = 2 * np.pi * j / num_theta
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        sphere_points.append([x, y, z])

sphere_points = np.array(sphere_points)

sphere_points = sphere_points.astype('float32')

# Define 3D points for a sphere
shifted_center = np.array([0.00125, 0.00125, 0.1])
centered_sphere_points = sphere_points - shifted_center

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    img = cv.resize(img, (new_width, new_height))

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        #line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        #line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        #cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        #cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
        #for b, t in zip(line_lower, line_upper):
            #cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Draw the sphere on the image
        
        projected_pts, _ = cv.projectPoints(centered_sphere_points, rvec, tvec, K, dist_coeff)
        projected_pts = np.int32(projected_pts).reshape(-1, 2)
        for pt in projected_pts:
            cv.circle(img, tuple(pt), 1, (255, 255, 255), -1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(1)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
