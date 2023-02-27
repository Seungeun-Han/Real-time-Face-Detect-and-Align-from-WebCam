# Real-time Face Detection and Alignment from WebCam

## Explain about this Repository

웹캠으로부터 영상을 가져와서 얼굴을 검출하는 코드입니다.

Face Detection을 위해 RetinaFace를 사용했으며, Face Alignment 방식은 Insightface 방식을 참고하였습니다.

추가적으로, 자동 밝기 및 대비 조절을 위해 __Gamma Correction 과 Constrast Normalization__ 연산을 추가하였습니다.


- [face_detect](#face_detect.py)
- [face_align](#face_align.py)
- [gammaCorrection_and_ContrastNormalization](#gammaCorrection_and_ContrastNormalization.py)
- [Reference](#Reference)

<hr>

## Explain about the codes respectively
## face_detect.py

- real-time으로 face를 detect하는 코드입니다..
- face와 left eye에 대한 bounding box 출력
- left/right eye, nose, left/right mouth 포인트 출력 (5 landmarks)
- detect 결과를 face_detect_result.jpg로 저장
- Retinaface, mobilenet0.25 사용

#### Result
![detect_result](https://user-images.githubusercontent.com/101082685/221504838-e0138952-3558-4099-8a74-8c2bca686a20.jpg)

<hr>

## face_align.py

- real-time으로 face를 align하는 코드입니다.
- insightface의 align 방식을 참고하였습니다.
- align 후 size는 __473X473__ 입니다.
- align 후엔 웹캠에서 읽어오는 얼굴이 기울어지거나, 멀거나 가까워져도 5개의 landmark의 위치는 변하지 않습니다.

#### Result
- Before Apply Alignment

![before_alignment_2](https://user-images.githubusercontent.com/101082685/221498721-cbbc6792-9475-449d-b945-d621ba274ee7.gif)

- After Apply Alignment

![after_alignment_3](https://user-images.githubusercontent.com/101082685/221498736-7fcee3d9-9d28-414a-9907-4a5d9aece0ba.gif)


얼굴이 기울어지거나 멀어져도 landmarks의 위치는 변하지 않는 것을 볼 수 있습니다.

<hr>

### gammaCorrection_and_ContrastNormalization.py

- real-time으로 영상을 보정하는 코드입니다.
- Gamma Correction 연산을 통해 Gamma 값을 계산하여 밝기를 적절하게 조정합니다.
- Contrast Normalization 연산을 통해 대비를 적절하게 조정합니다.

#### Result
- Before

![before_gc](https://user-images.githubusercontent.com/101082685/221508708-a5f3736f-c0c9-44e1-bba2-1ede7330ed33.png)


- After

![after_gc](https://user-images.githubusercontent.com/101082685/221508726-f16daba6-6947-4b76-ac5b-bcb98b1ee96f.png)

<hr>



## Reference

### Retinaface
- paper: https://arxiv.org/abs/1905.00641

- github:

  official: https://github.com/deepinsight/insightface

  what I used: https://github.com/biubug6/Pytorch_Retinaface


### face alignment
https://github.com/deepinsight/insightface



