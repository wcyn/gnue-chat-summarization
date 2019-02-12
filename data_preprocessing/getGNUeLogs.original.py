import requests
import codecs

filename = 'listOfUrlsForChatLogs.txt'
lines = open(filename).read().split('\n')

for url in lines:
    try:
        dateOfLog = url[-10:]
        print "processing", dateOfLog
        request = requests.get(url, allow_redirects=True)
        request = requests.get(request.url, allow_redirects=True)
        if request.history:
            request = requests.get(request.url, allow_redirects=True)
        html = request.content
        outfile = codecs.open("chatLogs/" + dateOfLog,"w")
        outfile.write(html)
        outfile.close()
    except Exception as e:
        print "\nCouldn't process: ", dateOfLog
        print "Error: ", e.message
        print "\n"

