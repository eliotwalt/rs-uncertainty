import re

with open("last_run.stdout.log") as f:
     started, done, running = set(), set(), set()
     for line in f.readlines():
         if "[" in line:
            num = int(re.findall(r"\d+", line)[2])
            if line.__contains__("] Processing id"): 
                started.add(num)
            elif line.__contains__("] Done processing id"): 
                done.add(num)
running = started - done
print("started:", list(sorted(list(started))))
print("done:", list(sorted(list(done))))
print("running:", list(sorted(list(running))))