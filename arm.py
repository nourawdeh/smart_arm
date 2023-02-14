import time
import os
from adafruit_servokit import ServoKit
import asyncio
import json
import speech_recognition as sr
import websockets
import requests

kit = ServoKit(channels=16)
thumb = None
index = None
middle = None
pinky = None


async def keep_connection(uri):
    async with websockets.connect(uri) as websocket:
        await websocket.send('Connected To Arm Successfully .')
        while True:
            response = await websocket.recv()
            try:
                response = json.loads(response)
                print(response)
                if response['command'] == 'move':
                    for action in response['sequence']:
                        await move_finger(finger_id=action['finger'], is_close=action['is_close'])
            except Exception as e:
                print(e)


async def move_finger(finger_id, is_close):
    global thumb, index, middle, pinky
    if is_close:
        angle = 180
    else:
        angle = 0
    if finger_id == 1:
        move_servo(thumb, angle)
        print(f'{angle} {finger_id}')
    if finger_id == 2:
        move_servo(index, angle)
        print(f'{angle} {finger_id}')
    if finger_id == 3:
        move_servo(middle, angle)
        print(f'{angle} {finger_id}')
    if finger_id == 4:
        move_servo(pinky, angle)
        print(f'{angle} {finger_id}')
    if finger_id == 5:
        move_servo(pinky, angle)
        print(f'{angle} {finger_id}')


async def recordAudioAndSend():
    while True:
        recognizer = sr.Recognizer()
        mic = sr.Microphone(device_index=1)
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print('Recording')
            # captured_audio = recognizer.record(source=source, duration=3)
            captured_audio = recognizer.listen(source, phrase_time_limit=5)
            print('End Of Record')
            print('Send Audio To Server')
            response = requests.post('http://127.0.0.1:8001/recognize', files={'record': captured_audio})
            #response = requests.post('http://127.0.0.1:8001/status', {'battery_percent': 90,up_time=3})
            print('Server Processed The Audio')
            print(response)


# function to move a servo motor to a specific angle
def move_servo(servo, angle):
    servo.angle = angle
    time.sleep(0.5)


async def main():
    global thumb, index, middle, pinky
    thumb = kit.servo[0]
    index = kit.servo[1]
    middle = kit.servo[2]
    pinky = kit.servo[3]
    recording_task = asyncio.create_task(recordAudioAndSend())
    connection_task = asyncio.create_task(keep_connection('ws://127.0.0.1:8001/ws'))
    await asyncio.gather(recording_task, connection_task)


asyncio.run(main())
