import json,time,urllib.request,sys
def ttft(content,mt=32):
    req=urllib.request.Request(chr(104)+chr(116)+chr(116)+chr(112)+chr(58)+chr(47)+chr(47)+chr(49)+chr(50)+chr(55)+chr(46)+chr(48)+chr(46)+chr(48)+chr(46)+chr(49)+chr(58)+chr(56)+chr(48)+chr(48)+chr(48)+chr(47)+chr(118)+chr(49)+chr(47)+chr(99)+chr(104)+chr(97)+chr(116)+chr(47)+chr(99)+chr(111)+chr(109)+chr(112)+chr(108)+chr(101)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115), data=json.dumps({chr(109)+chr(111)+chr(100)+chr(101)+chr(108):chr(109),chr(109)+chr(101)+chr(115)+chr(115)+chr(97)+chr(103)+chr(101)+chr(115):[{chr(114)+chr(111)+chr(108)+chr(101):chr(117)+chr(115)+chr(101)+chr(114),chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116):content}],chr(109)+chr(97)+chr(120)+chr(95)+chr(116)+chr(111)+chr(107)+chr(101)+chr(110)+chr(115):mt,chr(115)+chr(116)+chr(114)+chr(101)+chr(97)+chr(109):True}).encode(), headers={chr(67)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)+chr(45)+chr(84)+chr(121)+chr(112)+chr(101):chr(97)+chr(112)+chr(112)+chr(108)+chr(105)+chr(99)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(47)+chr(106)+chr(115)+chr(111)+chr(110)})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=300) as r:
        for line in r:
            if line.startswith(b"data:") and b"content" in line:
                return time.time()-t0
print("short TTFT= %.2f s" % ttft("你好"))
long_text="人工智能的发展历史可以追溯到20世纪50年代，图灵提出了著名的图灵测试。"*150
print("long(~5K tok) TTFT= %.2f s" % ttft(long_text+"请用一句话总结上文。"))