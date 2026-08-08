import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title='PPE Detection',
    page_icon='🦺',
    layout='wide'
)

st.title('🦺 PPE Detection System')

st.write(
    'Upload an image to detect helmets, safety vests, and safety shoes.'
)

@st.cache_resource
def load_model():
    return YOLO('best.pt')


model = load_model()

confidence = st.sidebar.slider(
    'Confidence Threshold',
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    'Upload an image',
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader('Original Image')
    st.image(image, use_container_width=True)

    with st.spinner('Detecting PPE...'):

        results = model.predict(
            source=image,
            conf=confidence
        )

    result = results[0]

    st.subheader('Detection Result')

    plotted_image = result.plot()

    st.image(
        plotted_image,
        channels='BGR',
        use_container_width=True
    )

    st.subheader('Detected Objects')

    if result.obb is not None and len(result.obb) > 0:

        for cls, conf in zip(
            result.obb.cls,
            result.obb.conf
        ):

            class_name = model.names[int(cls)]
            confidence_score = float(conf)

            st.write(
                f'**{class_name}** — '
                f'{confidence_score:.2%}'
            )

    else:

        st.warning('No PPE detected.')
