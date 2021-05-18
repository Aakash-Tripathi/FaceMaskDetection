import streamlit as st


def main():
    st.title('FACE MASK DETECTION')
    instructions = """
        Add Instructions here
        """
    st.write(instructions)
    file = st.file_uploader('')

    if file:
        st.image(file)
        st.success('Success message')


if __name__ == '__main__':
    main()
