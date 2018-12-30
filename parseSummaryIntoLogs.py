import io
import dateparser
import MySQLdb
import re

from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join

datasource_id = 48038
path = 'XMLsummaries'

# Open database connection
# db = MySQLdb.connect(
#     host="127.0.0.1",
#     user="root",
#     passwd="bubo1234",
#     db="gnue_irc",
#     use_unicode=True,
#     charset="utf8"
#     )
# cursor = db.cursor()
# cursor.execute('SET NAMES utf8mb4')
# cursor.execute('SET CHARACTER SET utf8mb4')
# cursor.execute('SET character_set_connection=utf8mb4')
files = [f for f in listdir(path) if isfile(join(path, f))][:1]
for filename in files:
    print('reading %s' % (filename))
    # xml = open(join(path,filename)).read()
    xml = io.open(join(path, filename), 'r', encoding='utf8')

    soup = BeautifulSoup(xml, features="html.parser")
    issueid = soup.issue.get('num')
    counter = 0
    autoincval = 0
    date_and_number = re.sub("[A-Za-z.]", "", filename)
    quote_date, quote_num = date_and_number.split('_')
    # Format quote date
    quote_date = quote_date[:4] + '-' + quote_date[4:6] + '-' + quote_date[6:]

    for sec in soup.find_all('section'):
        counter += 1
        title = sec.get('title')
        subject = sec.get('subject')
        archive = sec.get('archive')
        startdate = sec.get('startdate')
        startdate = dateparser.parse(startdate).date().isoformat()
        enddate = sec.get('enddate')
        enddate = dateparser.parse(enddate).date().isoformat()

        # Only store IRC chats
        if subject.startswith('[IRC]'):
            irc_date = subject.split('[IRC]')[1].strip()
            irc_date = dateparser.parse(irc_date).date().isoformat()

            '''
            print '==='
            print 'datasource', datasource_id
            print 'issueid', issueid
            print 'counter:',   counter
            print 'title:',    title
            print 'subject:',  subject
            print 'archive:',  archive
            print 'startdate:',startdate
            print 'enddate:',  enddate
            '''

            # try:
            #     cursor.execute(
            #         u"INSERT INTO GNUeSummaryItems(itemid, issueid, counter,\
            #         title, subject, archive, startdate, enddate, datasource_id) \
            #         VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s)",
            #         (
            #             issueid, counter, title, subject, archive, startdate,
            #             enddate, datasource_id
            #         )
            #     )
            #     db.commit()
            #     autoincval = cursor.lastrowid
            #
            # except MySQLdb.Error as error:
            #     print(error)
            #     db.rollback()
            #
            # for topic in sec.find_all('topic'):
            #     top = topic.string
            #
            #     try:
            #         cursor.execute(
            #             u"INSERT INTO GNUeSummaryTopics(topicid, itemid, topic) \
            #             VALUES (NULL, %s, %s)", (autoincval, top))
            #         db.commit()
            #
            #     except MySQLdb.Error as error:
            #         print(error)
            #         db.rollback()

            # for mention in sec.find_all('mention'):
            #     ment = mention.string
            #     mentu = ment.encode('utf-8').strip()
            #
            #     try:
            #         cursor.execute(
            #             u"INSERT INTO GNUeSummaryMentions(mentionid, itemid,\
            #             mention) VALUES (NULL, %s, %s)", (autoincval, mentu))
            #         db.commit()
            #
            #     except MySQLdb.Error as error:
            #         print(error)
            #         db.rollback()

            paracount = 0
            paraautoincval = 0
            summary_path = "summarizedLogs"
            summary_filename = irc_date + '.log.html'
            path_to_file = join(summary_path, summary_filename)
            html = io.open(path_to_file, 'r', encoding='utf8')
            log_soup = BeautifulSoup(html, "html.parser")
            table = log_soup.find('table')
            summary_div = log_soup.new_tag("div", **{'class':'summary'})
            # with open(path_to_file, "a") as f:
            #     f.write("<div class='summary'>")

            for p in sec.find_all('p'):

                paracount += 1
                # para = p.getreplace('\r\n', ' ')
                try:
                    # import pdb; pdb.set_trace()
                    print(irc_date)
                    # print(summary_div)
                    # print('\n')
                    summary_div.append(p)
                    # print('Para: %d. %s (%s) \n' %(paracount, para, irc_date))
                    # with open(path_to_file, "a") as f:
                    #     f.write(str(p))
                    # cursor.execute(
                    #     u"INSERT INTO GNUeSummaryPara(paraid, itemid, paracount,\
                    #     para, quote_date, issue_id) \
                    #     VALUES (NULL, %s, %s, %s, %s, %s)", (
                    #         autoincval, paracount, para, startdate, issueid
                    #         )
                    #     )
                    # db.commit()
                    # # get last used paragraph id (autoincrement)
                    # paraautoincval = cursor.lastrowid

                except Exception as error:
                    print(error)

            table.insert_after(summary_div)
            # print(log_soup.prettify())
            with open(path_to_file, "w") as f:
                f.write(str(log_soup))

            # with open(path_to_file, "a") as f:
            #     f.write("</div>")

                # for quote in p.find_all('quote'):
                #     who = quote.get('who')
                #     quo = quote.get_text().replace('\r\n', ' ')
                #
                #     try:
                #         cursor.execute(
                #             u"INSERT INTO GNUeSummaryParaQuotes(paraquoteid,\
                #             paraid, who, quote, quote_date, quote_num) \
                #             VALUES (NULL, %s, %s, %s, %s, %s)", (
                #                 paraautoincval, who, quo, irc_date, quote_num))
                #         db.commit()
                #
                #     except MySQLdb.Error as error:
                #         print(error)
                #         db.rollback()
# disconnect from server
# db.close()
