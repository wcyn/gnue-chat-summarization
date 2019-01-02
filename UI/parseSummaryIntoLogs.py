import io
import dateparser

from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join

datasource_id = 48038
path = 'XMLsummaries'

files = [f for f in listdir(path) if isfile(join(path, f))]
for filename in files:
    print('reading %s' % filename)
    xml = io.open(join(path, filename), 'r', encoding='utf8')

    soup = BeautifulSoup(xml, features="html.parser")
    issueid = soup.issue.get('num')

    for sec in soup.find_all('section'):
        title = sec.get('title')
        subject = sec.get('subject')
        archive = sec.get('archive')

        # Only store IRC chats
        if subject.startswith('[IRC]'):
            irc_date = subject.split('[IRC]')[1].strip()
            irc_date = dateparser.parse(irc_date).date().isoformat()

            paracount = 0
            summary_path = "summarizedLogs"
            summary_filename = irc_date + '.log.html'
            path_to_file = join(summary_path, summary_filename)
            html = io.open(path_to_file, 'r', encoding='utf8')
            log_soup = BeautifulSoup(html, "html.parser")
            table = log_soup.find('table')
            summary_div = log_soup.new_tag("div", **{'class': 'summary'})

            for p in sec.find_all('p'):

                paracount += 1
                try:
                    print(irc_date)
                    summary_div.append(p)

                except Exception as error:
                    print(error)

            table.insert_after(summary_div)
            with open(path_to_file, "w") as f:
                f.write(str(log_soup))
