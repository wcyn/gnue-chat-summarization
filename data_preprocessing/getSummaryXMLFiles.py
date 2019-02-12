# getSummaryXMLFiles.py
# fetches each XML file, using the master list of all file URLs

import urllib.request as request
import codecs

filename = 'listOfUrlsForSummariesXML.txt'
lines = open(filename).read().split('\n')

for url in lines:
    # chop off front of long URL
    dateOfXML = url[77:]
    print("processing", dateOfXML)
    try:
        xml = request.urlopen(url).read()
        outfile = codecs.open('XMLSummaries/' + dateOfXML, "w")
        outfile.write(xml)
        outfile.close()
    except request.URLError as e:
        print(e.reason)
