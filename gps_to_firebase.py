import serial
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pynmea2 # You might need to install this: pip install pynmea2

# --- Configuration ---
# Path to your Firebase service account key JSON file on your Raspberry Pi
SERVICE_ACCOUNT_KEY_PATH = 'your-project-id-firebase-adminsdk-xxxxx-xxxxxxxxxx.json' # REPLACE THIS
# Your Firebase Realtime Database URL
DATABASE_URL = 'https://your-project-id-default-rtdb.firebaseio.com/' # REPLACE THIS
# Serial port where your GPS module is connected (common for Raspberry Pi)
SERIAL_PORT = '/dev/ttyS0' # This is typically the hardware UART port on Raspberry Pi
# Baud rate for your GPS module (common rates: 9600, 4800, 115200)
BAUD_RATE = 9600 # REPLACE THIS if your GPS module uses a different baud rate
# Path in Firebase where location data will be stored
FIREBASE_LOCATION_PATH = 'devices/myRaspberryPi' # This ID should match what your web app listens to

# --- Firebase Initialization ---
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit()

# Get a reference to the Firebase Realtime Database
ref = db.reference(FIREBASE_LOCATION_PATH)

# --- Serial Port Initialization ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    print("Please ensure the GPS module is connected correctly and the serial port is enabled.")
    print("You might need to run `sudo raspi-config` to enable the serial hardware.")
    exit()

# --- Main Loop to Read GPS and Send to Firebase ---
def read_gps_and_send_to_firebase():
    print("Starting GPS data acquisition...")
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('$GPRMC') or line.startswith('$GNGGA'): # Look for RMC or GGA sentences
                try:
                    # pynmea2 is great for parsing NMEA sentences
                    msg = pynmea2.parse(line)
                    
                    # For $GPRMC: Recommended Minimum Specific GNSS Data
                    if isinstance(msg, pynmea2.types.talker.RMC) and msg.status == 'A': # 'A' means active/valid data
                        lat = msg.latitude
                        lon = msg.longitude
                        speed = msg.spd_over_grnd # knots
                        timestamp_utc = msg.timestamp # datetime.time object
                        
                        if lat != 0.0 and lon != 0.0: # Check for valid coordinates
                            location_data = {
                                'latitude': lat,
                                'longitude': lon,
                                'speed_knots': speed,
                                'timestamp_utc': str(timestamp_utc) # Convert time object to string
                            }
                            ref.set(location_data) # Update location in Firebase
                            print(f"Sent to Firebase: Lat={lat:.6f}, Lon={lon:.6f}, Speed={speed:.2f} knots")
                            time.sleep(5) # Send update every 5 seconds to avoid exceeding Firebase free tier limits quickly
                        else:
                            print(f"Invalid GPRMC data (lat/lon are 0): {line}")
                    
                    # For $GNGGA: GPS Fix Data (includes altitude and number of satellites)
                    elif isinstance(msg, pynmea2.types.talker.GGA):
                        lat = msg.latitude
                        lon = msg.longitude
                        altitude = msg.altitude # meters
                        num_satellites = msg.num_sv_used
                        
                        if lat != 0.0 and lon != 0.0:
                            # You can combine data from RMC and GGA if needed,
                            # For simplicity, we prioritize RMC as it's often more complete for tracking.
                            # If you only get GGA, uncomment and use this block:
                            # location_data = {
                            #     'latitude': lat,
                            #     'longitude': lon,
                            #     'altitude_m': altitude,
                            #     'num_satellites': num_satellites,
                            #     'timestamp_utc': str(msg.timestamp)
                            # }
                            # ref.set(location_data)
                            # print(f"Sent to Firebase (GGA): Lat={lat:.6f}, Lon={lon:.6f}, Alt={altitude:.2f}m")
                            # time.sleep(5)
                            pass # We're primarily using RMC if available for simple tracking
                        else:
                            print(f"Invalid GNGGA data (lat/lon are 0): {line}")
                            
                except pynmea2.ParseError as e:
                    print(f"NMEA Parse Error: {e} - Line: {line}")
                except Exception as e:
                    print(f"Error processing NMEA sentence: {e} - Line: {line}")
            else:
                pass # print(f"Received non-NMEA sentence: {line}") # Uncomment to see all serial output

        except serial.SerialException as e:
            print(f"Device disconnected or serial error: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(1) # Small delay before retrying

if __name__ == "__main__":
    read_gps_and_send_to_firebase()
