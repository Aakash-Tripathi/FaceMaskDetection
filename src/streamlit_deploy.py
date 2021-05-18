import streamlit as st


def main():
    st.title('FACE MASK DETECTION')
    instructions = """
        Add Instructions here
        """
    st.write(instructions)
    file = st.file_uploader('Upload An Image')
    st.title("Here is the image you've selected")
    st.title('How it works:')


if __name__ == '__main__':
    main()
