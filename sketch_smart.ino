#include<SoftwareSerial.h>  //시리얼통신을 위한 객체선언
#include<Servo.h>  //서보모터 객체선언
 Servo myservo1;  //서보모터 선언
 SoftwareSerial BTSerial(3,2);  //블루투스 송수신을 위한 핀설정
 byte strBuffer[2]; //데이터를 송신할 버퍼
 byte buffer[2];  //데이터를 수신받을 버퍼
 int bufferPosition; //버퍼에 데이터를 저장할때 기록할 위치 선언
 boolean temp = 0;  //문이 닫혔는지 확인하는 변수 설정
 byte buf[2];  //buf를 입력받을 배열선언

const int tenwon = 5;  //5번핀에 10원을 저장하기 위한 선언
const int fiftywon = 6;  //6번핀에 50원을 저장하기 위한 선언
const int hundwon = 7;  //7번핀에 100원을 저장하기 위한 선언
const int fivehundwon = 8; //8번핀에 500원을 저장하기 위한 선언

int total = 0; //총액을 위한 변수 total을 선언하고 초기값은 0으로 설정
int cnt = 0; //잠금장치에 대한 변수 초기값을 0으로 설정

void setup()
{
  Serial.begin(9600); //전송속도를 9600bps로 설정
  BTSerial.begin(9600);
  pinMode(tenwon,INPUT); //10원에 대한 입력 설정
  pinMode(fiftywon,INPUT); //50원에 대한 입력 설정
  pinMode(hundwon,INPUT); //100원에 대한 입력 설정
  pinMode(fivehundwon,INPUT); //500원에 대한 입력 설정
  myservo1.attach(9); //서보모터 시그널 핀 설정
  myservo1.write(90); //서보모터를 90도로 설정(open)
  delay(2000);  //잠금장치 딜레이를 2초로 설정
  myservo1.write(0); //서보모터를 0도로 설정(close)
  delay(2000); //잠금장치 딜레이를 2초로 설정
}

void loop()
{
  //블루투스 데이터 수신
   if(BTSerial.available()){ //블루투스로 데이터 수신
      byte data = BTSerial.read(); //수신 받을 데이터 저장
      Serial.write(data); //수신된 데이터를 시리얼 모니터로 출력
      buffer[bufferPosition++] = data; //수신받은 데이터를 버퍼에 저장

    if(data == '0'){ //블루투스를 통해 '0'이 입력되면 open
       Serial.write("open");  //open을 모니터로 출력
       if(data == '\n');{ //문자열 종료표시
          buffer[bufferPosition] = '\0'; 
          bufferPosition = 0;  //버퍼포지션을 0으로 설정
          BTSerial.flush();}  //시리얼 데이터가 전송완료까지 기다리는 함수
          myservo1.write(90); //서보모터를 90로 설정(open)
          cnt = 1; //잠금장치에 대한 변수를 1로 설정(open)

    }
     if(data == '1'){ //블루투스를 통해 '1'이 들어오면 Close
        Serial.write("Close");  //Close를 모니터로 출력
       if(data == '\n');{ //문자열 종료표시
          buffer[bufferPosition] = '\0'; 
          bufferPosition = 0;  //버퍼포지션을 0으로 설정
          BTSerial.flush();}  ////시리얼 데이터가 전송완료까지 기다리는 함수
          myservo1.write(0); //서보모터를 0로 설정(Close)
          cnt = 0; //잠금장치에 대한 변수를 0로 설정(Close)
          
     }
   }

  if(digitalRead(tenwon)==LOW){ //10원에 대한 디지털신호가 Low이면
    Serial.print(" tenwon : "); //tewnwon을 모니터로 출력
    total += 10;  //total 값에 10원을 더하여 저장
    Serial.println(total); //total값을 줄바꿈하여 모니터로 출력
    BTSerial.println(total);  //total값을 블루투스로 출력
    delay(100);  
  }
  if(digitalRead(fiftywon)==LOW){  //50원에 대한 디지털신호가 Low이면
    Serial.print(" fiftynwon : ");  //fiftynwon을 모니터로 출력
    total += 50;  //total 값에 50원을 더하여 저장
    Serial.println(total);  //total값을 줄바꿈하여 모니터로 출력
    BTSerial.println(total);   //total값을 블루투스로 출력
    delay(100);
  }
  if(digitalRead(hundwon)==LOW){  //100원에 대한 디지털신호가 Low이면
    Serial.print(" hundwon : ");  //hundwon을 모니터로 출력
    total += 100;  //total 값에 100원을 더하여 저장
    Serial.println(total);  //total값을 줄바꿈하여 모니터로 출력
    BTSerial.println(total);   //total값을 블루투스로 출력
    delay(100);
  }
  if(digitalRead(fivehundwon)==LOW){  //500원에 대한 디지털신호가 Low이면
    Serial.print(" fivehundwon : ");  //fivehundwon을 모니터로 출력
    total += 500;  //total 값에 500원을 더하여 저장
    Serial.println(total);  //total값을 줄바꿈하여 모니터로 출력
    BTSerial.println(total);   //total값을 블루투스로 출력
    delay(100);
  }
  
}
