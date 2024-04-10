# Camera_pose-estimation-and-AR
This program is a program that outputs phrases in AR on a chess board.


1. 카메라의 자세를 알아내서 유추한다. (카메라 캘리브레이션)
2. cv.findChessboardCorners() 함수를 사용하여 체스보드의 코너를 찾는다.
3. cv.solvePnP() 함수를 사용하여 체스보드 코너와 같은 실제 촬영한 이미지에서 발견된 2D 포인트와 이에 상응하는 실제 3D 포인트를 사용하여 카메라 자세를 추정한다.
4. 구의 좌표를 생성하고, 이를 출력한다.

