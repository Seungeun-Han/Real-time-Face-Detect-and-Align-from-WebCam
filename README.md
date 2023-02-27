# Face-Detect-and-Align-from-WebCam

## Explain about this Repository

웹캠으로부터 영상을 가져와서 얼굴을 검출하는 코드입니다.
Face Detection을 위해 RetinaFace를 사용했으며, Face Alignment 방식은 Insightface 방식을 참고하였습니다.

추가적으로, 자동 밝기 및 대비 조절을 위해 __Gamma Correction 과 Constrast Normalization__ 연산을 추가하였습니다.


## Explain about the codes respectively
### face_detect.py
- real-time으로 face를 detect하는 코드.
- face와 left eye에 대한 bounding box 출력
- left/right eye, nose, left/right mouth 포인트 출력 (5 landmarks)
- detect 결과를 face_detect_result.jpg로 저장
- Retinaface, mobilenet0.25 사용


![before_alignment_2](https://user-images.githubusercontent.com/101082685/221498721-cbbc6792-9475-449d-b945-d621ba274ee7.gif)
![after_alignment_3](https://user-images.githubusercontent.com/101082685/221498736-7fcee3d9-9d28-414a-9907-4a5d9aece0ba.gif)


## What's Difference?
  - Retinaface
  - Gamma correction
  - contrast normalizaiton


## Reference

### Retinaface
- paper: https://arxiv.org/abs/1905.00641

- github:

  official: https://github.com/deepinsight/insightface

  what I used: https://github.com/biubug6/Pytorch_Retinaface

### Gamma Correction
- gamma correction을 사용하여 drowsiness detection을 한 논문: https://www.mdpi.com/1999-5903/11/5/115

  > we applied gamma correction to enhance image contrast, which can help our method achieve better results,
  > 
  > noted by an increased accuracy of 2% in night environments.

- but, 논문에서 gamma 값을 명시하지 않아서 얼마로 할지 근거가 필요함.


### face normalization and recognition
- http://www1.cs.columbia.edu/~jebara/htmlpapers/UTHESIS/node46.html

- __face normalization이 얼굴 인식 분야에선 필요함.  but, face detection에선 꼭 필요할까? 에 대한 survey__

  paper: https://ieeexplore.ieee.org/document/5395980

  keypoint:

  >  Aim of the face detection system is to identify and locate all faces 
  >  
  >  regardless of their positions, scale, orientation, lighting conditions, expressions, etc

  > The recognition rate of 94 percent is obtained for original images whereas it increased to 97 percent for normalized images.



### face alignment
- https://github.com/1adrianb/face-alignment

- insightface의 face alignment 방식 사용 예정


### Face Image Normalization For Authentication Under Diverse Lighting Conditions 논문 참고

https://www.tdcommons.org/cgi/viewcontent.cgi?article=4488&context=dpubs_series
