import threading
import sys

f=open("/content/ogb/ogb/utils/d"+sys.argv[1])
lines=f.readlines()
c = -1


def out():
  threading.Timer(5.0, out).start()
  global lines, c
  c+=1
  print (lines[c].replace("\n", ""))




out()
