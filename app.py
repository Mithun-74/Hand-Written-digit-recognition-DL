import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

digits = [str(i) for i in range(10)]


def predict_digit(img):

    img = np.array(img)

    # grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize
    img_resize = cv2.resize(img_grey, (28, 28))

    # invert colors (important for MNIST)
    img_resize = cv2.bitwise_not(img_resize)

    # normalize
    newimg = tf.keras.utils.normalize(img_resize, axis=1)

    # reshape
    newimg = newimg.reshape(-1, 28, 28, 1)

    predictions = model.predict(newimg)[0]

    digit = np.argmax(predictions)

    return f"Predicted Digit: {digit}", dict(zip(digits, predictions))


with gr.Blocks() as demo:

    gr.Markdown(
        """
        # ✏️ Handwritten Digit Recognizer
        ### CNN-based MNIST Digit Classification
        Draw a digit or upload an image and the model will predict the number.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="numpy",
                label="Draw or Upload Digit",
                height=280,
                width=280
            )

            predict_btn = gr.Button("Predict Digit 🚀")

        with gr.Column():

            prediction_text = gr.Textbox(label="Prediction")

            confidence_chart = gr.Label(num_top_classes=3, label="Confidence")


    predict_btn.click(
        fn=predict_digit,
        inputs=image_input,
        outputs=[prediction_text, confidence_chart]
    )

    gr.Markdown(
        """
        ---
        **Model:** Convolutional Neural Network (CNN)  
        **Dataset:** MNIST Handwritten Digits  
        **Framework:** TensorFlow / Keras  
        **Project By:** MITHUN   
        """
    )


demo.launch(theme=gr.themes.Soft())