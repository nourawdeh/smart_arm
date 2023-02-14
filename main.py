import json
import os
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, Form, File, UploadFile, WebSocket
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import numpy as np
import speech_recognition as sr

app = FastAPI()
Base = declarative_base()
audio_path = 'waves'
ws = None
last_recorded_action = None
vectorizer = CountVectorizer()
model = None
encoder = None


class Action(BaseModel):
    id: int
    name: str
    created_at: str


class Status(BaseModel):
    id: int
    battery_percent: int
    up_time: int


class ActionORM(Base):
    __tablename__ = 'actions'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    sequence = Column(String)
    created_at = Column(String)


class StatusORM(Base):
    __tablename__ = 'status'
    id = Column(Integer, primary_key=True)
    battery_percent = Column(Integer)
    up_time = Column(Integer)


# Connect to SQLite database
engine = create_engine('sqlite:///smart_arm.db')
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws
    if not ws:
        ws = websocket
    await ws.accept()
    while True:
        data = await ws.receive_text()
        print(f"Arm says: {data}")
        actions = makeActionSequence(
            finger_1=True,
            finger_2=True,
            finger_3=True,
            finger_4=True,
            finger_5=True,
        )
        await ws.send_text(
            json.dumps({
                'command': 'move',
                'sequence': actions['sequence']
            })
        )


def makeActionSequence(finger_1=False, finger_2=False, finger_3=False, finger_4=False, finger_5=False):
    actions = {
        "sequence": [
            {
                "finger": 1,
                "is_close": finger_1
            },
            {
                "finger": 2,
                "is_close": finger_2
            },
            {
                "finger": 3,
                "is_close": finger_3
            },
            {
                "finger": 4,
                "is_close": finger_4
            },
            {
                "finger": 5,
                "is_close": finger_5
            }
        ]
    }
    return actions


@app.post('/recognize')
async def recognize(record: UploadFile = File(...)):
    global last_recorded_action, vectorizer, model, encoder, ws
    if not os.path.exists(audio_path):
        os.mkdir(audio_path)
    filename = audio_path + '/test.wave'
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(record.file, buffer)
        recognizer = sr.Recognizer()
    try:
        print('Recognize By Google')
        t = recognizer.recognize_google(record.file, language='ar-AR')
        print('Google Recognizing Done')
        f = open('text_file.txt', 'w', encoding='utf-8')
        f.writelines(t + '\n')
        f.close()
    except sr.UnknownValueError as U:
        print(U)
    except sr.RequestError as R:
        print(R)
        # Read the text file
    with open('text_file.txt', 'r', encoding='utf-8') as f:
        text_file = f.read()
    print(text_file)
    # Vectorize the text file
    text_file = vectorizer.transform([text_file])

    # Make a prediction
    predictions = model.predict(text_file)

    # Get the label for the predicted class
    predicted_label = encoder.inverse_transform(np.argmax(predictions, axis=True))
    print("The predicted label for the text file is: ", predicted_label)
    text_file = vectorizer.inverse_transform(text_file)
    text_file = ' '.join(text_file[0])
    text_file = text_file.encode('utf-8').decode('utf-8-sig')
    text_file = text_file.encode('utf-8-sig').decode('utf-8')
    if predicted_label == "فتح":
        if (
                "فتح" in text_file or "قتح الكف" in text_file or "الكف فتح" in text_file or "فتاح" in text_file or "افتحها" in text_file or "افتح الساعد" in text_file or "الساعد افتح" in text_file or "افتح الكف" in text_file or "الكف افتح" in text_file or "افتح كف" in text_file or "كف افتح" in text_file or "بسط الكف" in text_file or "الكف بسط" in text_file or "ابسط الكف" in text_file or "الكف ابسط" in text_file or "الرجاء بسط الكف" in text_file or "بسط الكف الرجاء" in text_file or "الرجاء الكف بسط" in text_file or "الكف الرجاء بسط" in text_file or "الكف بسط الرجاء" in text_file or "الرجاء ابسط الكف" in text_file or "الكف ابسط الرجاء" in text_file or "الكف ابسط الرجاء" in text_file or "الرجاء فتح" in text_file or "فتح الرجاء" in text_file or "رجاء فتح" in text_file or "فتح رجاء" in text_file or "رجاء بسط الكف" in text_file or "الكف بسط رجاء" in text_file or "بسط الكف رجاء" in text_file or "بسط رجاء الكف" in text_file or "رجاء افتح الكف" in text_file or "الكف افتح رجاء" in text_file or "الكف رجاء افتح" in text_file or "افتح الكف رجاء" in text_file or "افتح رجاء الكف" in text_file or "رجاء افتح الساعد" in text_file or "افتح الساعد رجاء" in text_file or "افتح رجاء الساعد" or "الساعد رجاء افتح" in text_file or "الساعد افتح رجاء" in text_file or "رجاء افتح الكف" in text_file or "افتح الكف رجاء" in text_file or "افتح رجاء الكف" in text_file or "الكف رجاء افتح" or "رجاء فتح الكف" in text_file or "الرجاء قم بفتح" in text_file or "الرجاء بفتح قم" in text_file or "قم بفتح الرجاء" in text_file or "قم الرجاء بفتح" in text_file or "بفتح الرجاء قم" in text_file or "رجاء بفتح قم" in text_file or "قم بفتح رجاء" in text_file or "قم رجاء بفتح" in text_file or "الرجاء افتح" in text_file or "رجاء افتح" in text_file or "افتح الأصابع" in text_file or "افتح اليد" in text_file or "فتح اليد" in text_file):
            print("فتح تم")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=False,
                finger_3=False,
                finger_4=False,
                finger_5=False,
            )
        else:
            print("فتح خطا")
    elif predicted_label == "غلق":
        if (
                "غلق" in text_file or "غلق الكف" in text_file or "الكف غلق" in text_file or "سكر" in text_file or "اغلق" in text_file):
            actions = makeActionSequence(
                finger_1=True,
                finger_2=True,
                finger_3=True,
                finger_4=True,
                finger_5=True,
            )
            print("تم اغلاق كف")
        else:
            print("خطا اغلاق كف")
    elif predicted_label == "غلق الابهام":
        if (
                "اغلاق ابهام" in text_file or "ابهام اغلاق" in text_file or "اغلاق الابهام" in text_file or "الابهام اغلاق" in text_file or "الرجاء اغلاق الابهام" in text_file or "الرجاء الابهام اغلاق" in text_file or "غلق الابهام" in text_file or "الابهام غلق" in text_file or "رجاء اغلق الابهام" in text_file or "رجاء الابهام اغلق" in text_file or "الابهام اغلق رجاء" in text_file or "الابهام رجاء اغلق" in text_file or "اغلق الابهام رجاء" in text_file or "اغلق رجاء الابهام" in text_file or "الرجاء غلق الابهام" in text_file or "الرجاء الابهام غلق" in text_file or "غلق الابهام الرجاء" in text_file or "غلق الرجاء الابهام" in text_file or "الابهام الرجاء غلق" in text_file or "الابهام غلق الرجاء" or "اغلق الابهام" in text_file or "الابهام اغلق" in text_file or "رجاء اغلاق الابهام" in text_file or "رجاء الابهام اغلاق" in text_file or "الابهام اغلاق رجاء" in text_file or "الابهام رجاء اغلاق" in text_file or "اغلاق رجاء الابهام" in text_file or "اغلاق الابهام رجاء" in text_file or "الابهام اغلاق رجاء" in text_file or "الابهام رجاء اغلاق" in text_file or "قم باغلاق الابهام" in text_file or "قم الابهام باغلاق" in text_file or "الابهام باغلاق قم" in text_file or "الابهام قم باغلاق" in text_file or "باغلاق قم الابهام" in text_file or "بالاغلاق الابهام قم" in text_file or "الرجاء قم باغلاق الابهام" in text_file or "الرجاء باغلاق الابهام قم" in text_file or "الرجاء الابهام قم بالاغلاق" in text_file or "قم بغلق الابهام" in text_file or "قم الابهام بغلق" in text_file or "الابهام بغلق قم" in text_file or "الابهام قم بغلق" or "بغلق قم الابهام" in text_file or "بغلق الابهام قم" or "رجاء قم باغلاق الابهام" in text_file or "رجاء باغلاق الابهام قم" in text_file or "رجاء الابهام قم باغلاق" in text_file):
            print("تم اغلاق ابهام")
            actions = makeActionSequence(
                finger_1=True,
                finger_2=False,
                finger_3=False,
                finger_4=False,
                finger_5=False,
            )
        else:
            print("اغلاق ابهام خطا")
    elif predicted_label == "فتح الابهام":
        if (
                "ابهام فتح" in text_file or "فتح ابهام" in text_file or "اوكي" in text_file or "تمام" in text_file or "فتح الابهام" in text_file or "الابهام فتح" in text_file or "الرجاء فتح الابهام" in text_file or "الرجاء الابهام فتح" in text_file or "الابهام فتح الرجاء" in text_file or "الابهام الرجاء فتح" in text_file or "فتح الرجاء الابهام" in text_file or "قتح الابهام الرجاء" in text_file or "رجاء قم بفتح الابهام" in text_file or "قم بفتح الابهام" in text_file or "قم الابهام بفتح" in text_file or "بفتح الابهام قم" in text_file or "بفتح قم الابهام" in text_file or "الابهام قم بفتح" in text_file or "الابهام بفتح قم" in text_file or "الرجاء افتح الابهام" in text_file or "الرجاء الابهام افتح" in text_file or "الابهام افتح الرجاء" in text_file or "الابهام الرجاء افتح" in text_file or "رجاء افتح الابهام" in text_file or "رجاء الابهام افتح" in text_file or "الابهام افتح رجاء" in text_file or "الابهام رجاء افتح" in text_file or "افتح رجاء الابهام" in text_file or "افتح الابهام رجاء" in text_file or "الرجاء قم بفتح الابهام" in text_file or "الرجاء بفتح الابهام قم" in text_file or "الرجاء الابهام قم بفتح" in text_file or "رجاء قم بفتح الابهام" in text_file or "رجاء بفتح الابهام قم" in text_file or "رجاء الابهام بفتح قم" in text_file or "افتح الابهام" in text_file or "الابهام افتح" in text_file or "فتاح الابهام" in text_file or "الابهام فتاح" in text_file or "فتح باهم" in text_file or "باهم فتح" in text_file):
            print("تم فتح ابهام")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=True,
                finger_3=True,
                finger_4=True,
                finger_5=True,
            )
        else:
            print("خطا فتح ابهام")
    elif predicted_label == "غلق السبابة":
        if (
                "السبابة سكر" in text_file or "سكر السبابة" in text_file or "اغلق الرجاء السبابة" in text_file or "الرجاء اغلق السبابة" in text_file or "الرجاء السبابة اغلق" in text_file or "السبابة اغلق الرجاء" in text_file or "السبابة الرجاء اغلق" in text_file or "اغلق الرجاء السبابة" in text_file or "اغلق السبابة الرجاء" in text_file or "اغلاق الرجاء السبابة" in text_file or "الرجاء اغلاق السبابة" in text_file or "اغلاق السبابة الرجاء" in text_file or "الرجاء السبابةاغلاق" in text_file or "غلق السبابة" in text_file or "السبابة غلق" in text_file or "رجاء اغلق السبابة" in text_file or "رجاء السبابة اغلق" in text_file or "اغلق رجاء السبابة" in text_file or "اغلق السبابة رجاء" in text_file or "السبابة رجاء اغلق" in text_file or "السبابة اغلق رجاء" in text_file or "الرجاء غلق السبابة" in text_file or "الرجاء السبابة غلق" in text_file or "السبابة الرجاء غلق" in text_file or "السبابة غلق الرجاء" in text_file or "غلق الرجاء السبابة" in text_file or "غلق السبابة الرجاء" in text_file or "اغلق السبابة" in text_file or "السبابة اغلق" in text_file or "رجاء اغلاق السبابة" in text_file or "رجاء السبابة اغلاق" in text_file or "السبابة اغلاق رجاء" in text_file or "السبابة رجاء اغلاق" in text_file or "اغلاق رجاء السبابة" in text_file or "اغلاق السبابة رجاء" in text_file or "اغلاق رجاء السبابة" in text_file or "السبابة اغلاق رجاء" in text_file or "قم باغلاق السبابة" in text_file or "السبابة باغلاق قم" in text_file or "الرجاء قم باغلاق السبابة" in text_file or "قم الرجاء باغلاق السبابة" in text_file or "السبابة باغلاق الرجاء قم" or "قم بغلق السبابة" in text_file or "بغلق قم السبابة" in text_file or "السبابة قم بغلق" in text_file or "السبابة بغلق قم" in text_file or "رجاء قم باغلاق السبابة" in text_file or "قم باغلاق السبابة رجاء" in text_file):
            print("تم غلق سبابة")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=True,
                finger_3=False,
                finger_4=False,
                finger_5=False,
            )
        else:
            print("خطا غلق سبابة")
    elif predicted_label == "فتح السبابة":
        if (
                "فتاح السبابة" in text_file or "السبابة فتاح" in text_file or "افتح السبابة" in text_file or "السبابة افتح" in text_file or "الرجاء افتح السبابة" in text_file or "الرجاء السبابة افتح" in text_file or "السبابة افتح الرجاء" or "السبابة الرجاء افتح" in text_file or "افتح الرجاء السبابة" in text_file or "افتح السبابة الرجاء" in text_file or "فتح السبابة" in text_file or "السبابة فتح" in text_file or "رجاء افتح السبابة" in text_file or "رجاء السبابة افتح" in text_file or "افتح السبابة رجاء" in text_file or "افتح رجاء السبابة" in text_file or "رجاء السبابة افتح" in text_file or "رجاء افتح السبابة" in text_file or "الرجاء فتح السبابة" in text_file or "الرجاء السبابة فتح" in text_file or "فتح السبابة الرجاء" in text_file or "فتح الرجاء السبابة" in text_file or "افتح السبابة" in text_file or "السبابة افتح" in text_file or "رجاء فتاح السبابة" in text_file or "رجاء السبابة فتاح" in text_file or "فتاح رجاء السبابة" in text_file or "السبابة فتاح رجاء" in text_file or "السبابة رجاء فتاح" in text_file or "قم بفتح السبابة" in text_file or "قم السبابة بفتح" in text_file or "السبابة قم بفتح" in text_file or "السبابة بفتح قم" in text_file or "بفتح السبابة قم" in text_file or "بفتح قم السبابة" in text_file or "الرجاء قم بفتح السبابة" in text_file or "الرجاء بفتح قم السبابة" in text_file or "الرجاء السبابة بفتح قم" in text_file or "الرجاء قم السبابة بفتح" or "قم بفتح السبابة" in text_file or "بفتح السبابة قم" in text_file or "قم السبابة بفتح" in text_file or "بفتح قم السبابة" in text_file or "رجاء قم بفتح السبابة" in text_file or "رجاء بفتح السبابة قم" in text_file or "رجاء بفتح قم السبابة"):
            print("تم فتح سبابة")
            actions = makeActionSequence(
                finger_1=True,
                finger_2=False,
                finger_3=True,
                finger_4=True,
                finger_5=True,
            )
        else:
            print("خطا فتح سبابة")
    elif predicted_label == "غلق الوسطى":
        if (
                "سكر الوسطى" in text_file or "الوسطى سكر" or "اغلاق الوسطى" in text_file or "الوسطى اغلاق" in text_file or "الرجاء اغلق الوسطى" in text_file or "الرجاء الوسطى اغلق" in text_file or "الوسطى اغلق الرجاء" in text_file or "الوسطى الرجاء اغلق" in text_file or "اغلق الوسطى الرجاء" in text_file or "اغلق الرجاء الوسطى" in text_file or "الرجاء اغلاق الوسطى" in text_file or "الرجاء الوسطى اغلاق" in text_file or "الوسطى اغلاق الرجاء" in text_file or "الوسطى الرجاء اغلاق" in text_file or "اغلاق الوسطى الرجاء" in text_file or "اغلاق الرجاء الوسطى" in text_file or "غلق الوسطى" in text_file or "الوسطى غلق" in text_file or "رجاء اغلق الوسطى" in text_file or "رجاء الوسطى اغلق" in text_file or "الوسطى اغلق رجاء" in text_file or "الوسطى رجاء اغلق" in text_file or "اغلق الوسطى رجاء" in text_file or "اغلق رجاء الوسطى" in text_file or "الرجاء غلق الوسطى" in text_file or "الرجاء الوسطى غلق" in text_file or "الوسطى غلق الرجاء" in text_file or "الوسطى الرجاء غلق" in text_file or "غلق الرجاء الوسطى" in text_file or "غلق الوسطى الرجاء" in text_file or "اغلق الوسطى" in text_file or "الوسطى اغلق" in text_file or "رجاء اغلاق الوسطى" in text_file or "اغلاق الوسطى رجاء" in text_file or "اغلاق رجاء الوسطى" in text_file or "رجاء الوسطى اغلاق" in text_file or "قم باغلاق الوسطى" in text_file or "باغلاق الوسطى قم" in text_file or "الوسطى باغلاق قم" in text_file or "الوسطى قم باغلاق" in text_file or "باغلاق قم الوسطى" in text_file or "الرجاء قم باغلاق الوسطى" in text_file or "قم بغلق الوسطى" in text_file or "رجاء قم باغلاق الوسطى" in text_file):
            print("تم غلق الوسطى")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=False,
                finger_3=True,
                finger_4=False,
                finger_5=False,
            )
        else:
            print("خطا غلق الوسطى")
    elif predicted_label == "فتح الوسطى":
        if (
                "فتاح الوسطى" in text_file or "الوسطى فتاح" in text_file or "الوسطى افتح" in text_file or "افتح الوسطى" in text_file or "الوسطى افتح" in text_file or "الرجاء افتح الوسطى" in text_file or "الرجاء الوسطى افتح" in text_file or "افتح الرجاء الوسطى" in text_file or "افتح الوسطى الرجاء" in text_file or "الوسطى الرجاء افتح" in text_file or "الوسطى افتح الرجاء" in text_file or "الرجاء فتح الوسطى" in text_file or "الرجاء الوسطى فتح" in text_file or "فتح الوسطى الرجاء" in text_file or "فتح الرجاء الوسطى" in text_file or "فتح الوسطى الرجاء" in text_file or "فتح الوسطى" in text_file or "الوسطى فتح" in text_file or "رجاء افتح الوسطى" in text_file or "رجاء الوسطى افتح" in text_file or "الوسطى افتح رجاء" in text_file or "الوسطى رجاء افتح" in text_file or "افتح رجاء الوسطى" in text_file or "افتح الوسطى رجاء" in text_file or "الرجاء فتح الوسطى" in text_file or "الرجاء الوسطى فتح" in text_file or "فتح الوسطى الرجاء" in text_file or "فتح الرجاء الوسطى" in text_file or "افتح الوسطى" in text_file or "رجاء افتاح الوسطى" in text_file or "قم بفتح الوسطى" in text_file or "الرجاء قم بفتح الوسطى" in text_file or "قم بفتح الوسطى" in text_file or "رجاء قم بفتح الوسطى" in text_file):
            print("تم فتح الوسطى")
            actions = makeActionSequence(
                finger_1=True,
                finger_2=True,
                finger_3=False,
                finger_4=True,
                finger_5=True,
            )
        else:
            print("خطا فتح الوسطى")
    elif predicted_label == "غلق البنصر":
        if (
                "سكر البنصر" in text_file or "البنصر سكر" in text_file or "اغلاق البنصر" in text_file or "البنصر اغلاق" in text_file or "الرجاء اغلق البنصر" in text_file or "الرجاء البنصر اغلق" in text_file or "البنصر الرجاء اغلق" in text_file or "البنصر اغلق الرجاء" in text_file or "اغلق البنصر الرجاء" in text_file or "اغلق الرجاء البنصر" in text_file or "الرجاء اغلاق البنصر" in text_file or "الرجاء البنصر اغلاق" in text_file or "البنصر الرجاء اغلاق" in text_file or "البنصر اغلاق الرجاء" in text_file or "اغلاق الرجاء البنصر" in text_file or "اغلاق البنصر الرجاء" in text_file or "غلق البنصر" in text_file or "البنصر غلق" in text_file or "رجاء اغلق البنصر" in text_file or "رجاء البنصر اغلق" in text_file or "البنصر اغلق رجاء" in text_file or "البنصر رجاء اغلق" in text_file or "اغلق رجاء البنصر" in text_file or "اغلق البنصر رجاء" in text_file or "الرجاء غلق البنصر" in text_file or "الرجاء البنصر غلق" in text_file or "البنصر غلق رجاء" in text_file or "البنصر رجاء غلق" in text_file or "غلق رجاء البنصر" in text_file or "غلق البنصر رجاء" in text_file or "اغلق البنصر" in text_file or "البنصر اغلق" in text_file or "رجاء اغلاق البنصر" in text_file or "رجاء البنصر اغلاق" in text_file or "البنصر رجاء اغلاق" in text_file or "البنصر اغلاق رجاء" in text_file or "اغلاق البنصر رجاء" in text_file or "اغلاق رجاء البنصر" in text_file or "قم باغلاق البنصر" in text_file or "باغلاق البنصر قم" in text_file or "باغلاق قم البنصر" in text_file or "الرجاء قم باغلاق البنصر" in text_file or "الرجاء باغلاق البنصر قم" in text_file or "قم بغلق البنصر" in text_file or "قم البنصر بغلق" in text_file or "بغلق البنصر قم" in text_file or "بغلق قم البنصر" in text_file or "رجاء قم باغلاق البنصر" in text_file):
            print("تم غلق البنصر")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=False,
                finger_3=False,
                finger_4=True,
                finger_5=False,
            )
        else:
            print("خطا غلق البنصر")
    elif predicted_label == "فتح البنصر":
        if (
                "فتاح البنصر" in text_file or "البنصر فتاح" in text_file or "فتح البنصر" in text_file or "البنصر فتح" in text_file or "الرجاء افتح البنصر" in text_file or "الرجاء البنصر افتح" in text_file or "افتح البنصر الرجاء" in text_file or "افتح الرجاء البنصر" in text_file or "البنصر الرجاء افتح" in text_file or "النبصر افتح الرجاء" in text_file or "الرجاء فتح البنصر" in text_file or "الرجاء البنصر فتح" in text_file or "البنصر فتح الرجاء" in text_file or "البنصر الرجاء فتح" in text_file or "فتح البنصر الرجاء" in text_file or "فتح الرجاء البنصر" in text_file or "فتح البنصر" in text_file or "البنصر فتح" in text_file or "رجاء افتح البنصر" in text_file or "رجاء البنصر افتح" in text_file or "البنصر افتح رجاء" in text_file or "البنصر رجاء افتح" in text_file or "افتح رجاء البنصر" in text_file or "افتح البنصر رجاء" in text_file or "الرجاء فتح البنصر" in text_file or "الرجاء البنصر فتح" in text_file or "البنصر فتح الرجاء" in text_file or "البنصر الرجاء فتح" in text_file or "فتح الرجاء البنصر" in text_file or "فتح البنصر الرجاء" in text_file or "افتح البنصر" in text_file or "البنصر افتح" in text_file or "رجاء افتح البنصر" in text_file or "رجاء البنصر افتح" in text_file or "البنصر افتح رجاء" in text_file or "البنصر رجاء افتح" in text_file or "افتح رجاء البنصر" in text_file or "افتح البنصر رجاء" in text_file or "قم بفتح البنصر" in text_file or "قم البنصر بفتح" in text_file or "البنصر بفتح قم" in text_file or "البنصر قم بفتح" in text_file or "بفتح قم البنصر" in text_file or "بفتح البنصر قم" in text_file or "الرجاء قم بفتح البنصر" in text_file or "قم بفتح البنصر" in text_file or "قم البنصر بفتح" in text_file or "بفتح البنصر قم" in text_file or "البنصر بفتح قم" in text_file or "رجاء قم بفتح البنصر" in text_file):
            print("تم فتح البنصر")
            actions = makeActionSequence(
                finger_1=True,
                finger_2=True,
                finger_3=True,
                finger_4=False,
                finger_5=True,
            )
        else:
            print("خطا فتح البنصر")
    elif predicted_label == "غلق الخنصر":
        if (
                "اغلاق الخنصر" in text_file or "الخنصر اغلاق" in text_file or "الرجاء اغلق الخنصر" in text_file or "الرجاء الخنصر اغلق" in text_file or "اغلق الخنصر الرجاء" in text_file or "اغلق الرجاء الخنصر" in text_file or "الخنصر الرجاء اغلق" in text_file or "الخنصر اغلق الرجاء" in text_file or "الرجاء اغلاق الخنصر" in text_file or "الرجاء الخنصر اغلاق" in text_file or "الخنصر الرجاء اغلاق" in text_file or "الخنصر اغلاق الرجاء" in text_file or "اغلاق الرجاء الخنصر" in text_file or "اغلاق الخنصر الرجاء" in text_file or "غلق الخنصر" in text_file or "الخنصر غلق" or "رجاء اغلق الخنصر" in text_file or "رجاء الخنصر اغلق" in text_file or "الخنصر اغلق الرجاء" in text_file or "الخنصر الرجاء اغلق" in text_file or "اغلق الخنصر الرجاء" in text_file or "اغلق الرجاء الخنصر" in text_file or "الرجاء غلق الخنصر" in text_file or "الرجاء الخنصر غلق" in text_file or "الرجاء اغلق الخنصر" in text_file or "اغلق الخنصر" in text_file or "رجاء اغلاق الخنصر" in text_file or "قم باغلاق الخنصر" in text_file or "الرجاء قم باغلاق الخنصر" in text_file or "قم بغلق الخنصر" in text_file or "الرجاء قم بغلق الخنصر" in text_file):
            print("تم غلق الخنصر")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=False,
                finger_3=False,
                finger_4=False,
                finger_5=True,
            )
        else:
            print("خطا غلق الخنصر")
    elif predicted_label == "فتح خنصر":
        if (
                "فتح خنصر" in text_file or "خنصر فتح" in text_file or "الرجاء افتح الخنصر" in text_file or "الرجاء فتح الخنصر" in text_file or "افتح الخنصر" in text_file or "الخنصر افتح" in text_file or "رجاء افتح الخنصر" in text_file or "الرجاء فتح الخنصر" in text_file or "الرجاء افتح الخنصر" in text_file or "افتح الخنصر" in text_file or "الخنصر افتح" in text_file or "رجاء افتح الخنصر" in text_file or "قم بفتح الخنصر" in text_file or "الرجاء قم بفتح الخنصر" in text_file or "قم بفتح الخنصر" in text_file or "الرجاء قم بفتح الخنصر" in text_file):
            print("تم فتح الخنصر")

            actions = makeActionSequence(
                finger_1=True,
                finger_2=True,
                finger_3=True,
                finger_4=True,
                finger_5=False,
            )
        else:
            print("خطا فتح الخنصر")
    elif predicted_label == "قلم":
        if (
                "رجاء مسيك قلم" in text_file or "رجاء قلم مسيك" in text_file or "مسيك قلم" in text_file or "قلم مسيك" in text_file or "قلم" in text_file or "مسك قلم" in text_file or "قلم مسك" in text_file or "امسك قلم" in text_file or "قلم امسك" in text_file or "رجاء قم بمسك القلم" in text_file or "مسيك القلم" in text_file or "القلم مسيك" in text_file or "رجاء مسيك القلم" in text_file or "الرجاء مسيك القلم" in text_file or "امسك القلم" in text_file or "القلم امسك" in text_file or "الرجاء قم بمسك القلم" in text_file or "الرجاء مسك القلم" in text_file or "مسك القلم" in text_file or "القلم مسك" in text_file or "يرجى مسك القلم" in text_file or "امسك القلم" in text_file or "القلم امسك" in text_file):
            print("تم قلم")
            actions = makeActionSequence(
                finger_1=True,
                finger_2=True,
                finger_3=True,
                finger_4=True,
                finger_5=True,
            )
        else:
            print("خطا قلم")
    elif predicted_label == "سلام":
        if (
                "افتح سبابة ووسطى" in text_file or "افتح السبابة والوسطى" in text_file or "افتح السبابه و الوسطى" in text_file or "فتح السبابة و الوسطى" in text_file or "افتح السبابه والوسطى" in text_file or "فتح السبابة و الوسطى" in text_file or "فتح السبابة والوسطى" in text_file or "فتح السبابه و الوسطى" in text_file or "فتح السبابه والوسطى" in text_file or "الرجاء فتح السبابة والوسطى" in text_file or "الرجاء افتح السبابة والوسطى" in text_file or "الرجاء قم بفتح السبابة والوسطى" in text_file or "فتاح السبابة و الوسطى" in text_file or "فتاح السبابه و الوسطى" in text_file or "فتاح السبابة والوسطى" in text_file or "قم بفتح السبابة والوسطى" in text_file or "سلام" in text_file or "سلم" in text_file):
            print("تم سلام")
            actions = makeActionSequence(
                finger_1=True,
                finger_2=False,
                finger_3=False,
                finger_4=False,
                finger_5=False,
            )
        else:
            print("خطا سلام")
    elif predicted_label == "ثلاث اصابع":
        if (
                "افتح السبابة والوسطى والبنصر" in text_file or "افتح السبابة والوسطى والبنصر" in text_file or "افتح السبابه و الوسطى و البنصر" in text_file or "فتح السبابة و الوسطى والبنصر" in text_file or "افتح السبابه والوسطى و البنصر" in text_file or "فتح السبابه والوسطى والبنصر" in text_file or "فتاح السبابة و الوسطى و البنصر" in text_file or "فتاح السبابة والوسطى والبنصر" in text_file or "فتاح السبابه والوسطى والبنصر" in text_file or "فتح السبابه و الوسطى و البنصر" in text_file or "الرجاء فتح السبابة والوسطى والبنصر" in text_file or "الرجاء افتح السبابة والوسطى والبنصر" in text_file or "الرجاء قم بفتح السبابة والوسطى والبنصر" in text_file or "فتاح السبابة والوسطى والبنصر" in text_file or "قم بفتح السبابة والوسطى والبنصر" in text_file or "قم بفتح السبابة و الوسطى و البنصر" in text_file or "فتاح السبابة والوسطى والبنصر" in text_file or "فتاح السبابة و الوسطى و البنصر" in text_file or "يرجى فتح السبابة والوسطى و البنصر" in text_file or "يرجى فتح السبابةوالوسطى والبنصر" in text_file or "يرجى فتح السبابة والوسطى و البنصر" in text_file or "ثلاث اصابع"):
            print("تم ثلاث اصابع")
            actions = makeActionSequence(
                finger_1=False,
                finger_2=False,
                finger_3=False,
                finger_4=True,
                finger_5=True,
            )
        else:
            print("خطا ثلاث اصابع")
    try:
        if ws is not None:
            await ws.send_text(
                json.dumps({
                    'command': 'move',
                    'sequence': actions
                })
            )
        return {
            'message': 'Command Sent Successfully .',
        }
    except Exception as e:
        return {
            'message': "Invalid JSON Sequence"
        }


@app.post('/command/sequence')
async def sequence(actions_sequence: str = Form(), record: UploadFile = File(...)):
    global last_recorded_action
    if not os.path.exists(audio_path):
        os.mkdir(audio_path)
    filename = audio_path + '/' + str(datetime.now()) + '.wave'
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(record.file, buffer)
    try:
        last_recorded_action = actions_sequence
        actions = json.loads(actions_sequence)['sequence']
        if ws is not None:
            await ws.send_text(
                json.dumps({
                    'command': 'move',
                    'sequence': actions
                })
            )
        return {
            'message': 'Command Sent Successfully .',
        }
    except Exception as e:
        return {
            'message': "Invalid JSON Sequence"
        }


@app.post("/actions/save")
async def save_last_sequence(action_name: str = Form()):
    global last_recorded_action
    if last_recorded_action is not None:
        session = SessionLocal()
        action_orm = ActionORM(name=action_name, sequence=last_recorded_action, created_at=datetime.now())
        session.add(action_orm)
        session.commit()
        session.refresh(action_orm)
        last_recorded_action = None
        return {
            'message': 'Action Saved Successfully .',
            'data': action_orm.__dict__
        }
    else:
        return {
            'message': 'No actions recorded yet',
        }


@app.get("/actions")
async def read_all_actions():
    session = SessionLocal()
    actions = session.query(ActionORM).all()
    if actions is None:
        return {
            'message': 'No Actions Found',
            'data': []
        }
    return {
        'message': 'Fetched successfully.',
        'data': [action.__dict__ for action in actions]
    }


@app.post("/actions/{id}")
async def apply_action(id: int):
    session = SessionLocal()
    action = session.query(ActionORM).filter(ActionORM.id == id).first()
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    else:
        try:
            if ws is not None:
                actions = json.loads(action.sequence)['sequence']
                await ws.send_text(
                    json.dumps({
                        'command': 'move',
                        'sequence': actions
                    })
                )
                return {
                    'message': 'Fetched successfully.',
                    'data': action.__dict__
                }
        except Exception as e:
            return {
                'message': e
            }


@app.post("/status")
async def create_status(battery_percent: str = Form(), up_time: str = Form()):
    session = SessionLocal()
    status_orm = StatusORM(battery_percent=battery_percent, up_time=up_time)
    session.add(status_orm)
    session.commit()
    session.refresh(status_orm)
    return {
        'message': 'Status Updated Successfully .',
        'data': status_orm.__dict__
    }


@app.get("/status")
async def read_status():
    session = SessionLocal()
    status = session.query(StatusORM).order_by(desc(StatusORM.id)).first()
    if status is None:
        return {
            'data': {
                "id": 0,
                "battery_percent": -1,
                "up_time": -1
            }
        }
    return {
        'data': status
    }


@app.on_event("startup")
async def startup_event():
    global vectorizer, model
    if os.path.exists(audio_path):
        shutil.rmtree(audio_path)
    df = pd.read_csv("dataset2.csv")
    print('Dataset Loaded')
    df["text"] = df["text"].str.replace("[^\w\s]", "")  # remove punctuation
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    onehotencoder = OneHotEncoder()
    y_train = onehotencoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_val = encoder.transform(y_val)
    y_val = onehotencoder.transform(y_val.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test)
    y_test = onehotencoder.transform(y_test.reshape(-1, 1)).toarray()
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(len(set(df["label"])), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=400, validation_data=(X_val, y_val))
    model.save("model.h5")
    scores = model.evaluate(X_test, y_test)
    print("Accuracy:", scores[1])
    f = open('text_file.txt', 'r+')
    f.truncate(0)
    print('Server Startup Done !')
