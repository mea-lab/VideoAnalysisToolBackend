from app.analysis.YOLOTracker import YOLOTracker
import json
import time

def write_output_to_file(output, file_path) :
    with open (file_path, 'w') as outfile:
        json.dump(output, outfile)

start_time = time.time()
ouputDict = YOLOTracker("rigidity_gaby.mp4",'yolov8n.pt','')
print("--- %s seconds ---" % (time.time() - start_time))

write_output_to_file(ouputDict, 'output.json')

