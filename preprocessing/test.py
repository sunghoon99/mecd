from mpu6050 import mpu6050
import time
import csv
import RPi.GPIO as GPIO

# MPU6050 센서 설정
mpu = mpu6050(0x68)

# GPIO 설정
buzzer_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

# CSV 파일 설정
csv_file = open('sensor_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["ax", "ay", "az", "gx", "gy", "gz", "label"])  # 헤더 작성

def read_sensor_data():
    # 센서에서 데이터 읽기
    accel_data = mpu.get_accel_data()
    gyro_data = mpu.get_gyro_data()
    return [accel_data['x'], accel_data['y'], accel_data['z'], gyro_data['x'], gyro_data['y'], gyro_data['z']]

try:
    while True:
        # 부저 울림
        GPIO.output(buzzer_pin, True)
        time.sleep(0.5)  # 부저가 0.5초 동안 울림
        GPIO.output(buzzer_pin, False)
        time.sleep(1)

        # 센서 데이터 읽기
        sensor_values = read_sensor_data()

        # 사용자에게 라벨 입력 받기
        label = input("Enter the label (squat or stool or stand): ")
        sensor_values.append(label)
        
        # CSV 파일에 데이터와 라벨 쓰기
        csv_writer.writerow(sensor_values)
        print("Data saved:", sensor_values)

        # 3초 간격으로 반복
        time.sleep(3 - 1.5)  # 이미 1.5초 지연되었으므로 추가로 1.5초 대기

except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    csv_file.close()
    GPIO.cleanup()

