import streamlit as st
from src.pred import prediction


def main():
    st.title('FACE MASK DETECTION')
    instructions = """
        ! Warning Grayscale images are not yet supported
        Please upload an image with only one person in it.
        """
    st.write(instructions)
    file = st.file_uploader('')

    if file:
        st.success('Image Uploaded')
        pred = prediction(file)
        st.write('PREDICTION:', pred)
        st.image(file)


if __name__ == '__main__':
    main()
