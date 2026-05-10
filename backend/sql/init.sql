CREATE DATABASE IF NOT EXISTS pedestrian_detection
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE pedestrian_detection;

CREATE TABLE IF NOT EXISTS users (
  id            INT UNSIGNED    NOT NULL AUTO_INCREMENT,
  username      VARCHAR(64)     NOT NULL,
  email         VARCHAR(128)    NOT NULL,
  password_hash VARCHAR(255)    NOT NULL,
  role          ENUM('admin','user') NOT NULL DEFAULT 'user',
  created_at    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_login    DATETIME        NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_username (username),
  UNIQUE KEY uq_email    (email)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS detection_jobs (
  id                  CHAR(36)     NOT NULL,
  user_id             INT UNSIGNED NOT NULL,
  input_type          ENUM('image','video','stream') NOT NULL,
  original_filename   VARCHAR(255) NULL,
  file_path           VARCHAR(512) NULL,
  status              ENUM('queued','processing','completed','failed') NOT NULL DEFAULT 'queued',
  model_version       VARCHAR(32)  NOT NULL DEFAULT 'enhanced-yolov8n-v1',
  conf_threshold      FLOAT        NOT NULL DEFAULT 0.25,
  iou_threshold       FLOAT        NOT NULL DEFAULT 0.45,
  pedestrian_count    INT          NULL,
  processing_time_ms  INT          NULL,
  error_message       TEXT         NULL,
  created_at          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at        DATETIME     NULL,
  PRIMARY KEY (id),
  CONSTRAINT fk_jobs_user FOREIGN KEY (user_id) REFERENCES users(id),
  INDEX idx_jobs_user_id  (user_id),
  INDEX idx_jobs_status   (status),
  INDEX idx_jobs_created  (created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS detection_results (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  job_id      CHAR(36)        NOT NULL,
  frame_index INT             NOT NULL DEFAULT 0,
  bbox_x1     FLOAT           NOT NULL,
  bbox_y1     FLOAT           NOT NULL,
  bbox_x2     FLOAT           NOT NULL,
  bbox_y2     FLOAT           NOT NULL,
  confidence  FLOAT           NOT NULL,
  class_id    TINYINT         NOT NULL DEFAULT 0,
  track_id    INT             NULL,
  PRIMARY KEY (id),
  CONSTRAINT fk_results_job FOREIGN KEY (job_id) REFERENCES detection_jobs(id)
    ON DELETE CASCADE,
  INDEX idx_results_job_frame (job_id, frame_index)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS job_audit_logs (
  id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  job_id     CHAR(36)        NOT NULL,
  event      VARCHAR(64)     NOT NULL,
  detail     JSON            NULL,
  created_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT fk_logs_job FOREIGN KEY (job_id) REFERENCES detection_jobs(id)
    ON DELETE CASCADE
) ENGINE=InnoDB;
