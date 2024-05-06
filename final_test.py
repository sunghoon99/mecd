from mpu6050 import mpu6050
import time
import RPi.GPIO as GPIO
from joblib import load
import numpy as np

# setting
mpu = mpu6050(0x68)
buzzer_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

# model, scaler load
model = load('model.joblib')
scaler = load('scaler.joblib')

# function

def read_sensor_data():
    accel_data = mpu.get_accel_data()
    gyro_data = mpu.get_gyro_data()
    return np.array([[accel_data['x'], accel_data['y'], accel_data['z'], gyro_data['x'], gyro_data['y'], gyro_data['z']]])

def predict_action(data):
    data_scaled = scaler.transform(data)
    return model.predict(data_scaled)[0]

def control_buzzer(action):
    if action == "stool":
        GPIO.output(buzzer_pin, True)
        time.sleep(0.1)
        GPIO.output(buzzer_pin, False)
        time.sleep(0.1)
        GPIO.output(buzzer_pin, True)
        time.sleep(0.1)
        GPIO.output(buzzer_pin, False)
    elif action == "squat":
        GPIO.output(buzzer_pin, True)
        time.sleep(0.3)
        GPIO.output(buzzer_pin, False)
    # 'stand'일 경우에는 부저가 울리지 않음

###### Test Section ######

try:
    while True:
        # 센서 데이터 읽기 및 예측
        new_data = read_sensor_data()
        prediction = predict_action(new_data)
        print("Predicted label:", prediction)

        # 부저 제어
        control_buzzer(prediction)

        time.sleep(0.5)  # 데이터 읽는 간격

except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    GPIO.cleanup()

