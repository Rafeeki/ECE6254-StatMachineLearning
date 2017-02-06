import urllib
import sys
from signal import signal, SIGPIPE, SIG_DFL
mypath = "http://mdav.ece.gatech.edu/ece-6254-spring2017/assignments/"
mylines = urllib.urlopen(mypath).readlines()
for item in mylines:
	if "href=" in item: 
		print item[item.index("href="):]
		#last = item[item.index("http"):item.find("\">")]
	#elif "./handout" in item: 
		#print item[item.index("./handout"):item.find("</A")] 
		#last = mypath + item[item.index("handout"):item.find("\">")]
#print last
#sys.stdout.write(last)
#sys.stdout.flush()
#signal(SIGPIPE,SIG_DFL)#&
