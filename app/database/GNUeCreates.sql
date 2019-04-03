-- phpMyAdmin SQL Dump
-- version 4.0.10deb1
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: May 25, 2015 at 12:31 PM
-- Server version: 5.5.38-0ubuntu0.14.04.1
-- PHP Version: 5.6.0-1+deb.sury.org~trusty+1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";
-- --------------------------------------------------------

--
-- Table structure for table `GNUeSummaryItems`
--

CREATE TABLE IF NOT EXISTS `GNUeSummaryItems` (
  `itemid` int(11) NOT NULL AUTO_INCREMENT,
  `issueid` int(11) DEFAULT NULL,
  `counter` int(11) DEFAULT NULL,
  `title` varchar(100) DEFAULT NULL,
  `subject` varchar(256) DEFAULT NULL,
  `archive` varchar(256) DEFAULT NULL,
  `startdate` varchar(45) DEFAULT NULL,
  `enddate` varchar(45) DEFAULT NULL,
  `datasource_id` int(11) NOT NULL,
  PRIMARY KEY (`itemid`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `GNUeSummaryMentions`
--

CREATE TABLE IF NOT EXISTS `GNUeSummaryMentions` (
  `mentionid` int(11) NOT NULL AUTO_INCREMENT,
  `itemid` int(11) DEFAULT NULL,
  `mention` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`mentionid`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `GNUeSummaryPara`
--

CREATE TABLE IF NOT EXISTS `GNUeSummaryPara` (
  `paraid` int(11) NOT NULL AUTO_INCREMENT,
  `itemid` int(11) DEFAULT NULL,
  `paracount` int(11) DEFAULT NULL,
  `para` varchar(8192) DEFAULT NULL,
  `quote_date` varchar(45) DEFAULT NULL,
  `issue_id` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`paraid`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `GNUeSummaryParaRaw`
--

CREATE TABLE IF NOT EXISTS `GNUeSummaryParaRaw` (
  `paraid` int(11) NOT NULL AUTO_INCREMENT,
  `itemid` int(11) DEFAULT NULL,
  `paracount` int(11) DEFAULT NULL,
  `para` varchar(8192) DEFAULT NULL,
  `quote_date` varchar(45) DEFAULT NULL,
  `issue_id` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`paraid`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `GNUeSummaryParaQuotes`
--

CREATE TABLE IF NOT EXISTS `GNUeSummaryParaQuotes` (
  `paraquoteid` int(11) NOT NULL AUTO_INCREMENT,
  `paraid` int(11) DEFAULT NULL,
  `who` varchar(128) DEFAULT NULL,
  `quote` varchar(8192) DEFAULT NULL,
  `quote_date` varchar(45) DEFAULT NULL,
  `quote_num` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`paraquoteid`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `GNUeSummaryTopics`
--

CREATE TABLE IF NOT EXISTS `GNUeSummaryTopics` (
  `topicid` int(11) NOT NULL AUTO_INCREMENT,
  `itemid` int(11) DEFAULT NULL,
  `topic` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`topicid`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4 ;


CREATE TABLE IF NOT EXISTS `GNUeIRCLogs` (
  `log_id` int(11) NOT NULL AUTO_INCREMENT,
  `line_count` varchar(45) DEFAULT NULL,
  `line_type` varchar(45) DEFAULT NULL,
  `send_user` varchar(128) DEFAULT NULL,
  `line_message` varchar(8192) DEFAULT NULL,
  `datasource_id` varchar(45) DEFAULT NULL,
  `date_of_log` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`log_id`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8mb4 ;

-- GNUeIRCLogs(date_of_log, line_count, type, send_user, line_message, datasource_id)

CREATE TABLE GNUeIRCLogs
(
  log_id                          INT AUTO_INCREMENT
    PRIMARY KEY,
  line_count                      VARCHAR(45)               NULL,
  line_type                       VARCHAR(45)               NULL,
  send_user                       VARCHAR(128)              NULL,
  line_message                    VARCHAR(8192)             NULL,
  datasource_id                   VARCHAR(45)               NULL,
  date_of_log                     VARCHAR(45)               NULL,
  is_summary                      TINYINT DEFAULT '0'       NOT NULL,
  username_color                  VARCHAR(8) DEFAULT '#ddd' NOT NULL,
  spelling_corrected_line_message TEXT                      NULL
)
  ENGINE = MyISAM;

CREATE INDEX date_of_log
  ON GNUeIRCLogs (date_of_log);

CREATE TABLE conversation_statistics
			(
			number_of_summaries INT DEFAULT 0 NOT NULL,
			number_of_true_predictions INT DEFAULT 0 NOT NULL,
			in_last_predicted_group BOOLEAN DEFAULT FALSE ,
			number_of_sentences INT DEFAULT 0 NOT NULL,
			conversation_date VARCHAR(64) PRIMARY KEY
			)

