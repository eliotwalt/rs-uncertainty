import sys, subprocess
import ee 
ee.Initialize(project=sys.argv[1])
tasks = ee.data.getTaskList()
cnt = 0
for task in tasks:
    if not task["state"] in ["CANCELLED", "COMPLETED", "FAILED"]:
        print("Cancelling task", task["id"])
        p = subprocess.Popen(f"earthengine task cancel {task['id']}".split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        cnt += 1
print(f"Cancelled {cnt} tasks")