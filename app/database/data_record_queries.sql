-- Select only logs with summaries (ones I have gone through
SELECT COUNT(*) FROM GNUeIRCLogs
WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18';
-- 578572 TOTAL Summarized
-- 20715 currently processed

SELECT log_id, line_message, date_of_log FROM GNUeIRCLogs
WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18'
ORDER BY log_id;


-- Generate chat date partitions. Filter by date to get partitions for only summarized chats

SELECT MIN(gnu.log_id) AS min_log_id,
  MAX(gnu.log_id) AS max_log_id,
  gnu.date_of_log, COUNT(gnu.log_id) as chat_line_count
FROM (SELECT log_id, date_of_log FROM GNUeIRCLogs) AS gnu
WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18'
GROUP BY gnu.date_of_log
ORDER BY MIN(gnu.log_id) ASC
;

SELECT * FROM GNUeIRCLogs where GNUeIRCLogs.log_id = 16183;

SELECT GNUeIRCLogs.line_message FROM GNUeIRCLogs;


SELECT log_id, is_summary FROM GNUeIRCLogs
-- WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18'
ORDER BY log_id ASC;


SELECT log_id, is_summary, prediction, date_of_log FROM GNUeIRCLogs
WHERE date_of_log in ('2001-11-04', '2001-11-15', '2001-10-26', '2001-11-01', '2001-10-30')
ORDER BY log_id ASC;


SELECT log_id, date_of_log, is_summary FROM GNUeIRCLogs
WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18'
ORDER BY log_id ASC;


SET SESSION group_concat_max_len = 100000;
SELECT date_of_log,
GROUP_CONCAT(
    log_id
    ORDER BY log_id ASC SEPARATOR ','
) AS log_ids_summary
FROM gnue_irc.GNUeIRCLogs
WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18'
GROUP BY date_of_log;
# UPDATE GNUeIRCLogs SET prediction=0;

INSERT INTO conversation_statistics (
conversation_date, number_of_true_predictions,
number_of_summaries, number_of_sentences)
VALUES ('2001-11-05', 0, 127, 2295)
ON DUPLICATE KEY UPDATE
number_of_true_predictions=VALUES(number_of_true_predictions),
number_of_summaries=VALUES(number_of_summaries),
number_of_sentences=VALUES(number_of_sentences);


-- DELETE FROM conversation_statistics;


SELECT * FROM GNUeIRCLogs
WHERE log_id=526850;


UPDATE GNUeIRCLogs
SET categorical_value_1 = 0, categorical_value_2 = 0
WHERE categorical_value_1 IS NULL OR categorical_value_2 IS NULL
;

SELECT * FROM conversation_statistics_2;
