import urllib.request as request
import codecs

filename = 'listOfUrlsForChatLogs.txt'
# lines = open(filename).read().split('\n')
lines = ['https://web.archive.org/web/20040517030110/http://www.gnuenterprise.org:80/irc-logs/gnue-public.log.2001-06-27']

for url in lines:
    dateOfLog = url[-10:]
    try:
        print("processing", dateOfLog)
        html = request.urlopen(url).read()
        outfile = codecs.open("chatLogs/new/" + dateOfLog,"w")
        outfile.write(html)
        outfile.close()
    except Exception as e:
        print("\nCouldn't process: ", dateOfLog)
        print("Error: ", str(e))
        print("\n")

