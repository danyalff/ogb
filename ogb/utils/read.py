# import threading
# import sys
# import base64

# f=open("/content/ogb/ogb/utils/d"+sys.argv[1])
# lines=f.readlines()
# c = -1


# def out():
#   threading.Timer(0.5, out).start()
#   global lines, c
#   c+=1
#   s = lines[c];
#   s = base64.b64decode(s)
#   s = str(s)
#   s = s.replace("\n", "")
#   print (s[2:len(s)-1])
 
# out()

import sched, time
import threading
import sys
import base64

f=open("/content/ogb/ogb/utils/d"+sys.argv[1])
lines=f.readlines()
c = -1

s = sched.scheduler(time.time, time.sleep)
def out(sc): 
  global lines, c
  c+=1
  try:
    st = lines[c];
    st = base64.b64decode(st)
    st = str(st)
    st = st.replace("\n", "")
    print (st[2:len(st)-1])
  except Exception as e:
     print(e)
    
  
    
  s.enter(0.1, 1, out, (sc,))

s.enter(0.1, 1, out, (s,))
s.run()
