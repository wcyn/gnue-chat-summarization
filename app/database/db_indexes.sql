show variables like 'myisam_sort_buffer_size';
-- myisam_sort_buffer_size | 8388608 -- DEFAULT VALUE
set global myisam_sort_buffer_size = 8589934592; -- 1GB


ALTER TABLE GNUeIRCLogs ADD INDEX (date_of_log);
-- mysql> ALTER TABLE GNUeIRCLogs ADD INDEX (date_of_log);
-- Query OK, 659165 rows affected (5 min 24.79 sec)
-- Records: 659165  Duplicates: 0  Warnings: 0

ALTER TABLE GNUeSummaryParaQuotes ADD INDEX (quote_date);
ALTER TABLE GNUeSummaryParaQuotes ADD INDEX (paraid);
CREATE INDEX GNUeSummaryParaQuotes_quote_date_index
  ON gnuesummaryparaquotes (quote_date);
