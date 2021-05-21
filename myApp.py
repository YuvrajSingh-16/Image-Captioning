import streamlit as st
import Caption_It
from PIL import Image

st.markdown('''
<link rel="preconnect" href="https://fonts.gstatic.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">

<div style=" font-family: 'Bebas Neue', cursive;">
    <h1 style='text-align: center; color: black;'>CAPTION IT</h1>
</div>
''', unsafe_allow_html=True)
st.markdown("")

st.markdown("<h3 style='text-align: center;'>Deep learning-based model which can generate Captions for a given image.</h3>", unsafe_allow_html=True)

#st.write("A Deep learning-based model which can generate Captions for a given image.")
@st.cache
def load_image(img_file):
    img_file = Image.open(img_file)
    return img_file

st.markdown("")
st.markdown("")
st.write("## Result Preview:")
sample_img = "./static/263854883_0f320c1562.jpg"
st.image(sample_img, height=250, width=300)
st.write("\"*The two small dogs run through the grass*\"")

st.markdown("")
st.markdown("")
st.write("## Upload Image:")
file = st.file_uploader("Upload your image to generate caption", type=['png', 'jpg', 'jpeg'])

if file is not None:
    ## Saving the file
    with open("./static/"+file.name, "wb") as f:
        f.write(file.getbuffer())
        
    img = load_image(file)
    st.image(img, height=250, width=300)

    encoded_img = Caption_It.encode_image("./static/"+file.name)
    caption = Caption_It.predict_caption(encoded_img)

    if caption:
        st.header("Predicted Caption :")
        st.write('"'+caption+'"')
        st.success("Caption Generated")
