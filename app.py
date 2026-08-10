import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from collections import Counter
import cv2
import tempfile
import os
import subprocess


# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title='PPE Safety Detection',
    page_icon='🦺',
    layout='wide'
)


# -----------------------------------
# CUSTOM DESIGN
# -----------------------------------

st.markdown(
    '''
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .status-safe {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-size: 20px;
        font-weight: 700;
    }

    .status-warning {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        font-size: 20px;
        font-weight: 700;
    }

    .status-danger {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-size: 20px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #888;
        padding-top: 40px;
        padding-bottom: 15px;
        font-size: 14px;
    }

    </style>
    ''',
    unsafe_allow_html=True
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return YOLO('best.pt')


model = load_model()


# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<div class="main-title">🦺 PPE Safety Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="subtitle">
    AI-powered PPE monitoring using YOLOv8 OBB to detect helmets,
    safety vests, and safety shoes.
    </div>
    ''',
    unsafe_allow_html=True
)


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title('⚙️ Detection Settings')

confidence_threshold = st.sidebar.slider(
    'Confidence Threshold',
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05
)

st.sidebar.markdown('---')

st.sidebar.markdown(
    '''
    ### Detectable PPE

    🪖 Helmet

    🦺 Safety Vest

    🥾 Safety Shoes
    '''
)

st.sidebar.markdown('---')

st.sidebar.caption(
    'Lower confidence may detect more objects but can increase false positives.'
)


# -----------------------------------
# UPLOAD IMAGE / VIDEO
# -----------------------------------

input_type = st.radio(
    'Choose input type:',
    ['🖼️ Image', '🎥 Video'],
    horizontal=True
)


# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

if input_type == '🖼️ Image':

    uploaded_file = st.file_uploader(
        '📤 Upload an image for PPE analysis',
        type=['jpg', 'jpeg', 'png']
    )


# -----------------------------------
# VIDEO UPLOAD
# -----------------------------------

else:

    uploaded_file = st.file_uploader(
        '🎥 Upload a video for PPE analysis',
        type=['mp4', 'avi', 'mov']
    )


# -----------------------------------
# NO FILE
# -----------------------------------

if uploaded_file is None:

    if input_type == '🖼️ Image':

        st.info('Upload an image to start PPE detection.')

    else:

        st.info('Upload a video to start PPE detection.')


# ===================================
# IMAGE DETECTION
# ===================================

elif input_type == '🖼️ Image':

    image = Image.open(uploaded_file).convert('RGB')


    # -----------------------------------
    # RUN DETECTION
    # -----------------------------------

    with st.spinner('🔍 Analyzing PPE...'):

        results = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )

    result = results[0]


    # -----------------------------------
    # EXTRACT DETECTIONS
    # -----------------------------------

    detected_classes = []
    detected_confidences = []

    if result.obb is not None and len(result.obb) > 0:

        for cls, conf in zip(
            result.obb.cls,
            result.obb.conf
        ):

            class_name = model.names[int(cls)]

            detected_classes.append(class_name)
            detected_confidences.append(float(conf))


    counts = Counter(detected_classes)

    helmet_count = counts.get('helmet', 0)
    vest_count = counts.get('safety_vest', 0)
    shoes_count = counts.get('safety_shoes', 0)

    total_detections = len(detected_classes)


    # -----------------------------------
    # METRICS
    # -----------------------------------

    st.markdown('## 📊 Detection Overview')

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        '🪖 Helmets',
        helmet_count
    )

    col2.metric(
        '🦺 Safety Vests',
        vest_count
    )

    col3.metric(
        '🥾 Safety Shoes',
        shoes_count
    )

    col4.metric(
        '🎯 Total Detections',
        total_detections
    )

    st.markdown('---')


    # -----------------------------------
    # IMAGES
    # -----------------------------------

    original_col, result_col = st.columns(2)

    with original_col:

        st.subheader('Original Image')

        st.image(
            image,
            use_container_width=True
        )


    with result_col:

        st.subheader('Detection Result')

        plotted_image = result.plot()

        st.image(
            plotted_image,
            channels='BGR',
            use_container_width=True
        )


    # -----------------------------------
    # STATUS
    # -----------------------------------

    st.markdown('## 🛡️ PPE Status')

    has_helmet = helmet_count > 0
    has_vest = vest_count > 0
    has_shoes = shoes_count > 0

    detected_types = sum(
        [
            has_helmet,
            has_vest,
            has_shoes
        ]
    )


    if detected_types == 3:

        st.markdown(
            '''
            <div class="status-safe">
            ✅ Complete PPE Set Detected
            </div>
            ''',
            unsafe_allow_html=True
        )


    elif detected_types > 0:

        st.markdown(
            '''
            <div class="status-warning">
            ⚠️ Partial PPE Detected
            </div>
            ''',
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            '''
            <div class="status-danger">
            ❌ No PPE Detected
            </div>
            ''',
            unsafe_allow_html=True
        )


    st.caption(
        'Status is based on PPE detected in the overall image, '
        'not PPE assigned to each individual person.'
    )


    # -----------------------------------
    # PPE CHECKLIST
    # -----------------------------------

    st.markdown('### PPE Checklist')

    check1, check2, check3 = st.columns(3)


    with check1:

        if has_helmet:

            st.success('🪖 Helmet Detected')

        else:

            st.error('🪖 Helmet Not Detected')


    with check2:

        if has_vest:

            st.success('🦺 Safety Vest Detected')

        else:

            st.error('🦺 Safety Vest Not Detected')


    with check3:

        if has_shoes:

            st.success('🥾 Safety Shoes Detected')

        else:

            st.error('🥾 Safety Shoes Not Detected')


    # -----------------------------------
    # DETECTION DETAILS
    # -----------------------------------

    st.markdown('## 🔎 Detection Details')


    if total_detections > 0:

        detection_data = []

        for index, (class_name, confidence) in enumerate(
            zip(
                detected_classes,
                detected_confidences
            ),
            start=1
        ):

            detection_data.append(
                {
                    'Detection': index,
                    'Class': class_name,
                    'Confidence': f'{confidence:.1%}'
                }
            )


        st.dataframe(
            detection_data,
            use_container_width=True,
            hide_index=True
        )


        highest_confidence = max(
            detected_confidences
        )


        st.metric(
            'Highest Confidence',
            f'{highest_confidence:.1%}'
        )


    else:

        st.warning(
            'No PPE objects were detected at the current confidence threshold.'
        )


    # -----------------------------------
    # DOWNLOAD IMAGE RESULT
    # -----------------------------------

    st.markdown('## 📥 Export Result')

    rgb_result = plotted_image[:, :, ::-1]

    result_pil = Image.fromarray(
        rgb_result
    )

    buffer = BytesIO()

    result_pil.save(
        buffer,
        format='JPEG',
        quality=95
    )

    st.download_button(
        label='⬇️ Download Detection Result',
        data=buffer.getvalue(),
        file_name='ppe_detection_result.jpg',
        mime='image/jpeg'
    )


# ===================================
# VIDEO DETECTION - BONUS
# ===================================

else:

    st.markdown('## 🎥 Video PPE Detection')


    # -----------------------------------
    # SAVE UPLOADED VIDEO
    # -----------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.mp4'
    ) as temp_input:

        temp_input.write(
            uploaded_file.read()
        )

        input_video_path = temp_input.name


    # -----------------------------------
    # CREATE OUTPUT VIDEO
    # -----------------------------------

    output_video_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.avi'
    ).name


    # -----------------------------------
    # OPEN VIDEO
    # -----------------------------------

    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    if fps <= 0:

        fps = 25


    fourcc = cv2.VideoWriter_fourcc(
        *'XVID'
    )

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )


    # -----------------------------------
    # PROCESS VIDEO
    # -----------------------------------

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    progress_bar = st.progress(0)

    status_text = st.empty()


    frame_count = 0


    with st.spinner('🎥 Analyzing video...'):

        while cap.isOpened():

            success, frame = cap.read()

            if not success:

                break


            # YOLO detection on current frame
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False
            )

            result = results[0]


            # Draw OBB detections
            plotted_frame = result.plot()


            # Write processed frame
            writer.write(
                plotted_frame
            )


            frame_count += 1


            if total_frames > 0:

                progress = min(
                    frame_count / total_frames,
                    1.0
                )

                progress_bar.progress(
                    progress
                )

                status_text.text(
                    f'Processing frame {frame_count} '
                    f'of {total_frames}'
                )


    cap.release()
    writer.release()


    progress_bar.progress(1.0)

    status_text.text(
        '✅ Video processing completed!'
    )

# -----------------------------------
    # CONVERT AVI TO MP4
    # -----------------------------------

    output_mp4_path = tempfile.NamedTemporaryFile(
    delete=False,
    suffix='.mp4'
    ).name

    subprocess.run(
    [
        'ffmpeg',
        '-y',
        '-i', output_video_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_mp4_path
    ],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
    )

    final_video_path = output_mp4_path
    
    # -----------------------------------
    # DISPLAY RESULT
    # -----------------------------------

    st.markdown('## 🎯 Detection Result')

    with open(
        final_video_path,
        'rb'
    ) as video_file:

        video_bytes = video_file.read()


    st.video(
        video_bytes
    )


    # -----------------------------------
    # DOWNLOAD VIDEO
    # -----------------------------------

    st.markdown('## 📥 Export Result')

    st.download_button(
        label='⬇️ Download Detection Video',
        data=video_bytes,
        file_name='ppe_detection_result.mp4',
        mime='video/mp4'
    )


    # -----------------------------------
    # CLEAN TEMP FILES
    # -----------------------------------

    try:

        os.remove(input_video_path)

        os.remove(output_video_path)

        if os.path.exists(output_mp4_path):

            os.remove(output_mp4_path)

    except Exception:

        pass


    # -----------------------------------
# FOOTER
# -----------------------------------

st.markdown(
    '''
    <div class="footer">
    PPE Safety Detection System • YOLOv8 OBB
    </div>
    ''',
    unsafe_allow_html=True
)