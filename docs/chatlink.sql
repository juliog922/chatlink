CREATE TABLE "whatsmeow_device" (
  "jid" TEXT PRIMARY KEY,
  "lid" TEXT,
  "facebook_uuid" uuid,
  "registration_id" BIGINT NOT NULL,
  "noise_key" bytea NOT NULL,
  "identity_key" bytea NOT NULL,
  "signed_pre_key" bytea NOT NULL,
  "signed_pre_key_id" INTEGER NOT NULL,
  "signed_pre_key_sig" bytea NOT NULL,
  "adv_key" bytea NOT NULL,
  "adv_details" bytea NOT NULL,
  "adv_account_sig" bytea NOT NULL,
  "adv_account_sig_key" bytea NOT NULL,
  "adv_device_sig" bytea NOT NULL,
  "platform" TEXT NOT NULL DEFAULT '',
  "business_name" TEXT NOT NULL DEFAULT '',
  "push_name" TEXT NOT NULL DEFAULT ''
);

CREATE TABLE "whatsmeow_identity_keys" (
  "our_jid" TEXT,
  "their_id" TEXT,
  "identity" bytea NOT NULL,
  PRIMARY KEY ("our_jid", "their_id")
);

CREATE TABLE "whatsmeow_pre_keys" (
  "jid" TEXT,
  "key_id" INTEGER,
  "key" bytea NOT NULL,
  "uploaded" BOOLEAN NOT NULL,
  PRIMARY KEY ("jid", "key_id")
);

CREATE TABLE "whatsmeow_sessions" (
  "our_jid" TEXT,
  "their_id" TEXT,
  "session" bytea,
  PRIMARY KEY ("our_jid", "their_id")
);

CREATE TABLE "whatsmeow_sender_keys" (
  "our_jid" TEXT,
  "chat_id" TEXT,
  "sender_id" TEXT,
  "sender_key" bytea NOT NULL,
  PRIMARY KEY ("our_jid", "chat_id", "sender_id")
);

CREATE TABLE "whatsmeow_app_state_sync_keys" (
  "jid" TEXT,
  "key_id" bytea,
  "key_data" bytea NOT NULL,
  "timestamp" BIGINT NOT NULL,
  "fingerprint" bytea NOT NULL,
  PRIMARY KEY ("jid", "key_id")
);

CREATE TABLE "whatsmeow_app_state_version" (
  "jid" TEXT,
  "name" TEXT,
  "version" BIGINT NOT NULL,
  "hash" bytea NOT NULL,
  PRIMARY KEY ("jid", "name")
);

CREATE TABLE "whatsmeow_app_state_mutation_macs" (
  "jid" TEXT,
  "name" TEXT,
  "version" BIGINT,
  "index_mac" bytea,
  "value_mac" bytea NOT NULL,
  PRIMARY KEY ("jid", "name", "version", "index_mac")
);

CREATE TABLE "whatsmeow_contacts" (
  "our_jid" TEXT,
  "their_jid" TEXT,
  "first_name" TEXT,
  "full_name" TEXT,
  "push_name" TEXT,
  "business_name" TEXT,
  PRIMARY KEY ("our_jid", "their_jid")
);

CREATE TABLE "whatsmeow_chat_settings" (
  "our_jid" TEXT,
  "chat_jid" TEXT,
  "muted_until" BIGINT NOT NULL DEFAULT 0,
  "pinned" BOOLEAN NOT NULL DEFAULT false,
  "archived" BOOLEAN NOT NULL DEFAULT false,
  PRIMARY KEY ("our_jid", "chat_jid")
);

CREATE TABLE "whatsmeow_message_secrets" (
  "our_jid" TEXT,
  "chat_jid" TEXT,
  "sender_jid" TEXT,
  "message_id" TEXT,
  "key" bytea NOT NULL,
  PRIMARY KEY ("our_jid", "chat_jid", "sender_jid", "message_id")
);

CREATE TABLE "whatsmeow_privacy_tokens" (
  "our_jid" TEXT,
  "their_jid" TEXT,
  "token" bytea NOT NULL,
  "timestamp" BIGINT NOT NULL,
  PRIMARY KEY ("our_jid", "their_jid")
);

CREATE TABLE "whatsmeow_lid_map" (
  "lid" TEXT PRIMARY KEY,
  "pn" TEXT UNIQUE NOT NULL
);

CREATE TABLE "whatsmeow_event_buffer" (
  "our_jid" TEXT NOT NULL,
  "ciphertext_hash" bytea NOT NULL,
  "plaintext" bytea,
  "server_timestamp" BIGINT NOT NULL,
  "insert_timestamp" BIGINT NOT NULL,
  PRIMARY KEY ("our_jid", "ciphertext_hash")
);

CREATE TABLE "users" (
  "id" SERIAL PRIMARY KEY,
  "phone" TEXT UNIQUE NOT NULL,
  "email" TEXT UNIQUE NOT NULL,
  "name" TEXT,
  "role" TEXT NOT NULL
);

CREATE TABLE "clients" (
  "id" SERIAL PRIMARY KEY,
  "phone" TEXT UNIQUE NOT NULL,
  "email" TEXT UNIQUE NOT NULL,
  "name" TEXT
);

CREATE TABLE "products" (
  "ref" TEXT PRIMARY KEY,
  "description" TEXT,
  "price" NUMERIC(10,2) NOT NULL,
  "category" TEXT
);

CREATE TABLE "messages" (
  "id" SERIAL PRIMARY KEY,
  "client_id" INTEGER NOT NULL,
  "client_phone" TEXT NOT NULL,
  "user_id" INTEGER,
  "user_phone" TEXT NOT NULL
  "direction" TEXT NOT NULL,
  "type" TEXT NOT NULL,
  "content" TEXT,
  "timestamp" TIMESTAMP NOT NULL
);

CREATE INDEX ON "messages" ("client_id");

CREATE INDEX ON "messages" ("user_id");

ALTER TABLE "whatsmeow_identity_keys" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_pre_keys" ADD FOREIGN KEY ("jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_sessions" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_sender_keys" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_app_state_sync_keys" ADD FOREIGN KEY ("jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_app_state_version" ADD FOREIGN KEY ("jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_app_state_mutation_macs" ADD FOREIGN KEY ("jid", "name") REFERENCES "whatsmeow_app_state_version" ("jid", "name") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_contacts" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_chat_settings" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_message_secrets" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "whatsmeow_event_buffer" ADD FOREIGN KEY ("our_jid") REFERENCES "whatsmeow_device" ("jid") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "messages" ADD FOREIGN KEY ("client_id") REFERENCES "clients" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "messages" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON UPDATE SET NULL;
