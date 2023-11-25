import os
from FaceRecognitionRpi.face_recognizer import recog_func
from FaceRecognitionRpi.datagenerator import student_images
from FaceRecognitionRpi.face_trainer import imageTrainer
from pyfiglet import figlet_format

## Title Bar
def title():
    os.system('cls')
    figlet = figlet_format('OpenSourcePundits', font='digital')
    print(figlet)
    print(40*"/" + 40*'\\')
    print("***** Face Recognition Based Attendance System And Temperature Monitoring *****")
    print(40*"\\" + 40*'/')


def main_menu():
    title()
    print("")
    print(10* '*', "Welcome Student", 10 * '*')
    print("Select Your Choice")
    print("[USER INPUT][1] Collect New Faces")
    print("[USER INPUT][2] Train Images of Students")
    print("[USER INPUT][3] Collect Attendance and Measure Temperature")
    print("[USER INPUT][4] Send the Mail")
    print("[USER INPUT][5] Check For Mask")

    while True:
        try:
            user_choice = int(input("Please Enter Your Choice-->"))


            if user_choice == 1:
                collect_images()
                break

            elif user_choice == 2:
                image_trainer()
                break

            elif user_choice == 3:
                allot_attendance()
                break

            elif user_choice == 4:
                print("Initiating Email Notification Procedure")
                os.system("py mail.py")
                break
            
            else:
                print("Invalid Choice")
                main_menu()

        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")






def collect_images():
    student_images()
    print("Initiating Data Collection Process!")
    exit_key = input("Press any Key to retrun to Main Menu")
    main_menu()

def image_trainer():
    print("Initiating Training Image Program")
    imageTrainer()
    exit_key = input("Press any Key to retrun to Main Menu")
    main_menu()

def allot_attendance():
    print("Initating Automatic Attendace Marking System!")
    recog_func()
    exit_key = input("Press any Key to retrun to Main Menu")
    main_menu()

## Calling the Main Funtion
if __name__ == '__main__':
    main_menu()















