import imaplib
m = imaplib.IMAP4_SSL("imap.gmail.com")
m.login("bot@kapalua.es", "olwxsuvbyjogbcox")  # or your IMAP user/app-password
typ, data = m.list()
print("\n".join(x.decode(errors="ignore") for x in data))
m.logout()