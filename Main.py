import json
import socket
import bisect

import av
import cv2
from datetime import datetime
from multiprocessing import Process, Manager

FRAME_OFFSET = 450
FULL_SEI_UID = 1272825341085959061573

# localIP    = "192.168.31.144"

localDJIRTSPPort = 8554
localUDPPort = 10004

bufferSize = 1024
# bufferSize = 65504


class DroneData:
    def __init__(self, battery, gpsCount, alt, lat, lng, facing):
        self.battery = battery
        self.gpsCount = gpsCount
        self.alt = alt
        self.lat = lat
        self.lng = lng
        self.facing = facing


# def insertDataToDict(d: dict, pts, droneData):
#     d[pts] = droneData
#
# def popDataFromDict(d: dict, pts) -> DroneData:
#     return d.pop(pts, None)


def extractSEIFromPacket(avPacket: av.packet.Packet) -> list[bytes]:
    sei_units = []
    data = bytes(avPacket)

    i = 0
    while i < len(data) - 4:
        # Look for H.264 start codes: 0x00000001
        if data[i:i+4] == b'\x00\x00\x00\x01':
            i += 4

            nalType = data[i] & 0x1F; i += 1
            # print("NAL type: ", nalType)
            if nalType != 6: # NAL type for SEI metadata
                continue

            seiType = data[i] & 0x1F; i += 1
            # print("SEI type: ", seiType)
            if seiType != 5: # SEI type for user defined data
                continue

            sei_start = i

            seiSize = data[i]
            sei_end = i + 1 + 16 + seiSize
            i = sei_end
            # print("data[sei_end]: ", data[i])

            if data[i] != 0x80:
                i += 1
                continue

            sei_units.append(data[sei_start:sei_end])  # strip NAL header, NAL type, and SEI type bytes
        else:
            i += 1
    return sei_units


def convertDegreeToWindDirection(degree: float) -> str:
    if degree < -180 or degree > 180:
        return "-"

    elif -157.5 <= degree < -112.5:
        return "SW"

    elif -112.5 <= degree < -67.5:
        return "W"

    elif -67.5 <= degree < -22.5:
        return "NW"

    elif -22.5 <= degree < 22.5:
        return "N"

    elif 22.5 <= degree < 67.5:
        return "NE"

    elif 67.5 <= degree < 112.5:
        return "E"

    elif 112.5 <= degree < 157.5:
        return "SE"

    else:
        return "S"


def findPTS(sortedList, key):
    """Find rightmost value less than or equal to x"""
    i = bisect.bisect_right(sortedList, key)
    if i:
        return sortedList[i - 1]
    return None


def listenUDP(droneDataDict: dict, udpServerSocket: socket):
    # change to JSON
    print("UDP server up and listening {}", udpServerSocket)

    while True:
        try:
            bytesAddressPair = udpServerSocket.recvfrom(bufferSize)

            message = bytesAddressPair[0].decode('utf-8')
            # address = bytesAddressPair[1]

            jsonData = json.loads(message)

            if 'pts' in jsonData:
                presentationTimeMs = jsonData['pts']
                battery = jsonData['battery']
                gpsCount = jsonData['gpsCount']
                alt = jsonData['alt']
                lat = jsonData['lat']
                lng = jsonData['lng']
                facing = jsonData['facing']

                data = DroneData(battery, gpsCount, alt, lat, lng, facing)
                droneDataDict[presentationTimeMs] = data

                logMsg = "{} {} Drone Data:{}".format(datetime.now(), jsonData['pts'], jsonData)
                print(logMsg)

        except KeyboardInterrupt:
            break


def listenRTSP(droneDataDict: dict, rtspURL):
    print("RTSP URL: {}".format(rtspURL))

    container = av.open(rtspURL, 'r')
    video_stream = next((s for s in container.streams if s.type == 'video' and s.codec.name == 'h264'), None)
    # Ensure we do not auto-seek or drop timestamps

    video_stream.thread_type = 'AUTO'
    currDataPTS = 0
    currData: DroneData = DroneData(0, 0, 0, 0, 0, 0)

    for packet in container.demux(video_stream):
        seiPTS = 0
        dataChanged = False

        sei_list = extractSEIFromPacket(packet)
        for sei in sei_list:
            pos = 0

            size = sei[0]; pos += 1
            uuid1 = int.from_bytes(sei[pos:(pos + 8)], byteorder="big", signed=False); pos += 8
            uuid2 = int.from_bytes(sei[pos:(pos + 8)], byteorder="big", signed=False); pos += 8

            # Decode if uuid meets criteria

            # if uuid1 == FULL_SEI_UID:
            if uuid1 == 69 and uuid1 == uuid2:
                seiPTS = (int.from_bytes(sei[pos:(pos + size)], byteorder="big", signed=False)) - FRAME_OFFSET

                sortedDataPTS = sorted(droneDataDict.keys())
                newDataPTS = findPTS(sortedDataPTS, seiPTS)
                # print(f"Data PTS: {newDataPTS} [{seiPTS}]")

                if currDataPTS != newDataPTS and newDataPTS is not None:
                    newData = droneDataDict.pop(newDataPTS)
                    if currDataPTS <= newDataPTS:
                        currData = newData
                        currDataPTS = newDataPTS
                        dataChanged = True

                print(f"{datetime.now()} {currDataPTS}[{'Y' if dataChanged else 'N'}] - Data: {vars(currData)}")

        try:
            frames = packet.decode()
        except Exception:
            continue  # drop current frame on decoding error

        for frame in frames:
            img = frame.to_ndarray(format='bgr24')

            # Show the frame and timestamp
            timestamp_text = [f"SEI PTS: {seiPTS}",
                              f"Curr PTS: {currDataPTS}",
                              f"battery: {'-' if currData is None else currData.battery}",
                              f"gpsCount: {'-' if currData is None else currData.gpsCount}",
                              f"alt: {0.0 if currData is None else float(currData.alt)}",
                              f"lat: {0.0 if currData is None else float(currData.lat)}",
                              f"lng: {0.0 if currData is None else float(currData.lng)}",
                              f"facing: {'-' if currData is None else float(currData.facing)}",
                              f"[{'-' if currData is None else convertDegreeToWindDirection(float(currData.facing))}]"]

            y0 = 30
            for i in range(0, len(timestamp_text)):
                y = y0 * (i + 1)
                cv2.putText(img, timestamp_text[i], (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Test", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()



def main():
    manager = Manager()
    mainDict = manager.dict()

    # Create a datagram socket
    udpIP = input("Enter this PC's IP addr: ")

    rtspIP = udpIP
    rtspUsername = input("Enter RTSP username: ")
    rtspPassword = input("Enter RTSP password: ")


    # Bind to local address and ip to UDP socket
    udpServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    udpServerSocket.bind((udpIP, localUDPPort))

    # Listen for incoming datagrams
    rtspURL = f"rtsp://{rtspUsername}:{rtspPassword}@{rtspIP}:{localDJIRTSPPort}/djistream"
    # rtspURL = f"rtsp://localhost:{localDJIRTSPPort}/djistream"


    # loop = asyncio.get_event_loop()
    # asyncio.set_event_loop(loop)
    # loop.create_task(listenUDP(mainDict, udpServerSocket))
    # loop.create_task(listenRTSP(mainDict, rtspURL))
    # loop.run_forever()

    pUDP = Process(target=listenUDP, args=(mainDict, udpServerSocket))
    pRTSP = Process(target=listenRTSP, args=(mainDict, rtspURL))

    pUDP.start()
    pRTSP.start()

    # Setup RTSP
    pUDP.join()
    pRTSP.join()


if __name__ == '__main__':
    main()
