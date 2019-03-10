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
WHERE date_of_log >= '2001-10-23' AND date_of_log <= '2001-11-18'
ORDER BY log_id ASC;


1259 + 439 + 448 + 1132 + 255 + 481 + 653 + 393 + 191 + 604 + 220 + 893 + 627 + 2043 + 800 + 1990 + 1051 + 536 + 1053 + 1348
