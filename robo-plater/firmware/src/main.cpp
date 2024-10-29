#include <Arduino.h>
#include <SPI.h>

// SPI settings
#define SPI_CLOCK_SPEED 1000000 // 1 MHz
#define SPI_MODE SPI_MODE0
#define SPI_BIT_ORDER MSBFIRST

// SPI pins
// SPI pins based on ESP32-S3 FSPI default pinout
const int SCK_PIN = 12;   // FSPI SCK
const int MOSI_PIN = 11;  // FSPI MOSI 
const int MISO_PIN = 13;  // FSPI MISO
const int CS_PIN = 10;

// Register address
#define CONTROL_REGISTER_ADDRESS 0x00

// Function declarations
void writeControlRegister(uint8_t value);
void displayControlRegisterStatus(uint8_t value);
uint8_t readControlRegister();



void setup() {
    USBSerial.begin(115200);
    // Initialize CS pin
    pinMode(CS_PIN, OUTPUT);
    digitalWrite(CS_PIN, HIGH); // Deselect the device

    // Initialize SPI
    SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, CS_PIN);
    SPI.setDataMode(SPI_MODE);
    SPI.setBitOrder(SPI_BIT_ORDER);
    SPI.setClockDivider(SPI_CLOCK_DIV16); // Adjust as needed

    USBSerial.println("Starting CONTROL register manipulation...");

    // Initially disable LEDs
    writeControlRegister(0x00);
}

void loop() {
    // Enable LED1 and LED2
    writeControlRegister(0x03); // Set bits 0 and 1 to 1

    USBSerial.println("LED1 and LED2 enabled.");
    uint8_t controlRegValue = readControlRegister();
    displayControlRegisterStatus(controlRegValue);

    delay(5000); // Keep LEDs on for 5 seconds

    // Disable LED1 and LED2
    writeControlRegister(0x00); // Clear bits 0 and 1

    USBSerial.println("LED1 and LED2 disabled.");
    controlRegValue = readControlRegister();
    displayControlRegisterStatus(controlRegValue);

    delay(5000); // Keep LEDs off for 5 seconds
}

// Function to read the CONTROL register
uint8_t readControlRegister() {
    uint8_t readAddress = CONTROL_REGISTER_ADDRESS | 0x80; // Set MSB for read operation
    uint8_t receivedData = 0;

    digitalWrite(CS_PIN, LOW); // Select the device
    SPI.transfer(readAddress); // Send read command
    receivedData = SPI.transfer(0x00); // Receive the data
    digitalWrite(CS_PIN, HIGH); // Deselect the device

    return receivedData;
}

// Function to write to the CONTROL register
void writeControlRegister(uint8_t value) {
    uint8_t writeAddress = CONTROL_REGISTER_ADDRESS & 0x7F; // Clear MSB for write operation

    digitalWrite(CS_PIN, LOW); // Select the device
    SPI.transfer(writeAddress); // Send write command
    SPI.transfer(value); // Send data
    digitalWrite(CS_PIN, HIGH); // Deselect the device
}

// Function to display the status of the CONTROL register
void displayControlRegisterStatus(uint8_t controlRegValue) {
    // Parse CONTROL register bits
    bool led1Enabled = controlRegValue & 0x01;
    bool led2Enabled = (controlRegValue >> 1) & 0x01;
    bool vled1SampleEnabled = (controlRegValue >> 2) & 0x01;
    bool vled2SampleEnabled = (controlRegValue >> 3) & 0x01;
    bool thermSampleEnabled = (controlRegValue >> 4) & 0x01;

    // Display the status
    USBSerial.println("CONTROL Register Status:");
    USBSerial.print("  LED1 Enabled: "); USBSerial.println(led1Enabled ? "Yes" : "No");
    USBSerial.print("  LED2 Enabled: "); USBSerial.println(led2Enabled ? "Yes" : "No");
    USBSerial.print("  VLED1 Sampling Enabled: "); USBSerial.println(vled1SampleEnabled ? "Yes" : "No");
    USBSerial.print("  VLED2 Sampling Enabled: "); USBSerial.println(vled2SampleEnabled ? "Yes" : "No");
    USBSerial.print("  Thermal Sampling Enabled: "); USBSerial.println(thermSampleEnabled ? "Yes" : "No");
    USBSerial.println();
}