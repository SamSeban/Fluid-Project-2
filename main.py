import cv2
import numpy as np
from matplotlib import pyplot as plt

clicks = 0

def select_point(event, x, y, flags, param):
    global point_selected, top, bottom, clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if (clicks == 0):
            top = (x, y)
        if (clicks == 1):
            bottom = (x, y)
            point_selected = True
            cv2.destroyAllWindows()
        clicks += 1

def show_timer(time_elapsed):
    timer_window = np.zeros((200, 400), dtype=np.uint8)
    text = f"{int(time_elapsed // 60):02d}:{int(time_elapsed % 60):02d},{int((time_elapsed % 1) * 100):02d}"
    
    # Adjust the font scale based on the window size
    window_height, window_width = timer_window.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 5

    while True:
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        if text_width < window_width * 0.9 and text_height < window_height * 0.9:
            font_scale += 0.1
        else:
            font_scale -= 0.1
            break

    # Calculate the position of the text so it's centered
    x = (window_width - text_width) // 2
    y = (window_height + text_height) // 2

    cv2.putText(timer_window, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imshow('Timer', timer_window)


def filter_colors(frame, color_str):
    # Convert RGB string to BGR
    r, g, b = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
    color_bgr = np.uint8([[[b, g, r]]])

    # Convert frame and color to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    
    # Create mask
    hue = color_hsv[0][0]
    lower_bound = np.array([max(0, hue[0] - 40), max(0, hue[1] - 127), max(0, hue[2] - 127)])
    upper_bound = np.array([min(179, hue[0] + 40), min(255, hue[1] + 127), min(255, hue[2] + 127)])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result

def removeEnd(theo):
    # Find the index where the data begins to increase
    diff = np.diff(theo)
    increasing_index = np.where(diff > 0)[0]
    if increasing_index.size > 0:
        increasing_index = increasing_index[0] + 1
        # Set values to 0 after the index
        theo[increasing_index:] = 0
    return theo

def find_top_non_black_point(image, y_start, y_end):
    startIndex = int(y_start + (1/2)*(y_end - y_start))
    non_black_indices = np.argwhere(np.any(image[startIndex:], axis=-1))
    if non_black_indices.size > 0:
        return non_black_indices[0][::-1] + [0, startIndex]
    return None

def track_point(filename, totalHeight, liquid_color):
    global point_selected, top, bottom, clicks
    point_selected = False
    top = ()
    bottom = ()
    heights = []
    oldY = 0

    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()

    if not ret:
        print("Error reading video file.")
        return

    frame_resized = cv2.resize(frame, (720, 720))
    filtered_frame = filter_colors(frame_resized, liquid_color)

    cv2.namedWindow('Point tracking selector')
    cv2.setMouseCallback('Point tracking selector', select_point)

    while not point_selected:
        frame_with_text = frame_resized.copy()
        if (clicks == 0):
            cv2.putText(frame_with_text, 'Select Top of Cup', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Point tracking selector', frame_with_text)
        if (clicks == 1):
            cv2.putText(frame_with_text, 'Select Bottom of Cup', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Point tracking selector', frame_with_text)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not point_selected:
        print("No point selected.")
        return

    cap = cv2.VideoCapture(filename)

    top = np.array([[[top[0], top[1]]]], dtype=np.float32)
    bottom = np.array([[[bottom[0], bottom[1]]]], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (720, 720))
        filtered_frame = filter_colors(frame_resized, liquid_color)

        yT = top.ravel()[1]
        cv2.line(frame_resized, (0, int(yT)), (720, int(yT)), (7, 217, 237), 5)

        yB = bottom.ravel()[1]
        cv2.line(frame_resized, (0, int(yB)), (720, int(yB)), (7, 217, 237), 5)

        y = find_top_non_black_point(filtered_frame,int(yT) + 1, int(yB))
        if y is None: 
            y = oldY
        else:
            y = y[1]
        oldY = y
        cv2.line(frame_resized, (0, int(y)), (720, int(y)), (140, 241, 249), 5)

        height = yT - yB
        currentHeight = (float(y - yB) / height) * float(totalHeight)

        heights.append(currentHeight)

        cv2.imshow('Tracked Point', frame_resized)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        show_timer(current_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return heights

if __name__ == '__main__':
    cupHeight = 65.0 # [mm] Height of the cup
    cupDiameter = 68.91 # [mm] Diameter of the cup
    cupVolume = (np.pi * (cupDiameter / 2)**2 * cupHeight) * 10**(-9);  # [m^3]

    T_f = 298 # [K] Final temperature
    
    P_atm = 101325 # [Pa] => [kg/(m*s^2)]
    rho = 997 # [kg/m^3]
    g = 9.81 # [m/s^2]

    filename = input("Enter the video file path: ")
#    liquid_color = input("Enter the RGB color of the liquid: ")
    liquid_color = "923F00" # Color of the water
    heights = np.array(track_point(filename, cupHeight, liquid_color)) # [cm]

    time = np.arange(len(heights)) # Time in frames
    time = time / 30 # Time in seconds

    dpi = 96  # Display's DPI
    width_in_inches = 800 / dpi
    height_in_inches = 700 / dpi

    plt.figure(1, figsize=(width_in_inches, height_in_inches))
    plt.figure(1)
    plt.subplots_adjust(wspace=0.5, hspace=0.45, bottom=0.2)

    # HEIGHT
    plt.subplot(2, 2, 1)
    plt.plot(time, heights)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Height [mm]')
    plt.title('Height v. Time')
    plt.legend(["Experimental value"])

    # PRESSURE
    heightsMeter = heights * 0.001 # [mm] => [m]
    z = heightsMeter - heightsMeter[0]
    pressure = P_atm-rho*g*z # [Pa]
    pressureKpa = pressure/1000 # [kPa]

    plt.subplot(2, 2, 3)

    plt.plot(time, pressureKpa)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [kPa]')
    plt.title('Pressure v. Time')
    plt.legend(["Resulting Pressure"])

    # TEMPERATURE
    V_water = (np.pi * (cupDiameter / 2)**2 * heights) * 10**(-9) # [m^3]
    volume = cupVolume - V_water

    V_water_end = (np.pi * (cupDiameter / 2)**2 * heights[-1]) * 10**(-9)
    V_f = cupVolume - V_water_end

    temperature = ((pressure*volume)/(pressure[-1]*V_f)) * T_f
    temperatureCelsius = temperature - 273.15 # [°C]

    plt.subplot(2, 2, 4)
    plt.plot(time, temperatureCelsius)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [°C]')
    plt.title('Temperature v. Time')
    plt.legend(["Resulting Temperature"])

    # VOLUME
    plt.subplot(2, 2, 2)
    plt.plot(time, volume*1000) # [L]
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Volume [L]')
    plt.title('Volume of air in the cup v. Time')
    plt.legend(["Experimental value"])

    plt.show()