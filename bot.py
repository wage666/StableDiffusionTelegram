import torch
from torchvision import models
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from io import BytesIO
import random

load_dotenv()
TG_TOKEN = os.getenv('YOUR_TOKEN_HERE')
MODEL_GENERATION = 'shahidul034/text_generation_bangla_model'

LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
revision = "fp16" if LOW_VRAM_MODE else None
torch_dtype = torch.float16 if LOW_VRAM_MODE else None

# Load the image generation pipeline
generationPipeline = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_GENERATION, revision=revision, torch_dtype=torch_dtype)
generationPipeline = generationPipeline.to("cuda")

# Load the DeepLab model
model_segmentation = models.segmentation.deeplabv3_resnet101(pretrained=True)
model_segmentation = model_segmentation.to("cuda")
model_segmentation.eval()

# Disable safety checker if wanted
def dummy_checker(images, **kwargs):
    return images, False

# Image to bytes conversion function
def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio

# Function to get the "Try again" and "Variations" inline keyboard markup
def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup

# Generate and send photo based on text input
async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)

    prompt = update.message.text
    
    # Perform segmentation on the input image
    segmentation_result = perform_segmentation(prompt)
    
    # Apply the segmentation mask to the generated image
    generated_image = generationPipeline(prompt=prompt)
    generated_image = generated_image * segmentation_result
    
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(generated_image), caption=f'"{prompt}"', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

# Generate and send photo based on photo input
async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)

    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()

    # Perform segmentation on the input photo
    segmentation_result = perform_segmentation(photo)

    # Apply the segmentation mask to the generated image
    generated_image = generationPipeline(photo=photo)
    generated_image = generated_image * segmentation_result

    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(generated_image), reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

# Perform segmentation on the input image
def perform_segmentation(image):
    input_tensor = preprocess_image(image).to("cuda")
    with torch.no_grad():
        output = model_segmentation(input_tensor)["out"]
    masks = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return masks

# Preprocess the input image
def preprocess_image(image):
    image = Image.open(BytesIO(image))
    # Add preprocessing steps if needed
    return image

# Handle button callbacks
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message

    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_message(update.effective_user.id, "Sorry, image generation variations are not available.")

app = ApplicationBuilder().token(TG_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo))
app.add_handler(MessageHandler(filters.PHOTO, generate_and_send_photo_from_photo))
app.add_handler(CallbackQueryHandler(button))

app.run_polling()
