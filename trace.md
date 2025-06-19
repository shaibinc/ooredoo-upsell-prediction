2025-06-19T12:34:23.180525007Z **\_**
2025-06-19T12:34:23.180564000Z / \_ \ ****\_\_**** ****\_**** \_**_
2025-06-19T12:34:23.180568939Z / /_\ \\\_** / | \_ ** \_/ ** \
2025-06-19T12:34:23.180573257Z / | \/ /| | /| | \/\ **\_/
2025-06-19T12:34:23.180578236Z \_\_**|** /\_\_\_** \_**\_/ |**| \_** >
2025-06-19T12:34:23.180584768Z \/ \/ \/
2025-06-19T12:34:23.180590649Z A P P S E R V I C E O N L I N U X
2025-06-19T12:34:23.180596280Z
2025-06-19T12:34:23.180602341Z Documentation: http://aka.ms/webapp-linux
2025-06-19T12:34:23.180607881Z Python 3.9.22
2025-06-19T12:34:23.180614093Z Note: Any data outside '/home' is not persisted
2025-06-19T12:34:23.585432983Z Starting OpenBSD Secure Shell server: sshd.
2025-06-19T12:34:23.615014330Z WEBSITES_INCLUDE_CLOUD_CERTS is not set to true.
2025-06-19T12:34:23.640767446Z Updating certificates in /etc/ssl/certs...
2025-06-19T12:34:28.375796281Z 3 added, 0 removed; done.
2025-06-19T12:34:28.376824240Z Running hooks in /etc/ca-certificates/update.d...
2025-06-19T12:34:28.381054264Z done.
2025-06-19T12:34:28.386946045Z CA certificates copied and updated successfully.
2025-06-19T12:34:28.424210831Z Site's appCommandLine: startup.sh
2025-06-19T12:34:28.427588202Z App command line is a file on disk
2025-06-19T12:34:28.433251879Z App will launch in debug mode
2025-06-19T12:34:28.438787478Z Launching oryx with: create-script -appPath /home/site/wwwroot -output /opt/startup/startup.sh -virtualEnvName antenv -defaultApp /opt/defaultsite -userStartupCommand startup.sh -debugAdapter ptvsd -debugPort 49494
2025-06-19T12:34:29.621450913Z Found build manifest file at '/home/site/wwwroot/oryx-manifest.toml'. Deserializing it...
2025-06-19T12:34:29.623565338Z Build Operation ID: 9c08ca15c916fb5a
2025-06-19T12:34:29.625038925Z Output is compressed. Extracting it...
2025-06-19T12:34:29.625208901Z Oryx Version: 0.2.20250505.1, Commit: bec89959884a7663432d6e3c4d36acc30657cb85, ReleaseTagName: 20250505.1
2025-06-19T12:34:29.626251806Z Extracting '/home/site/wwwroot/output.tar.gz' to directory '/tmp/8ddaf2d5f5ba34f'...
2025-06-19T12:34:34.833637793Z App path is set to '/tmp/8ddaf2d5f5ba34f'
2025-06-19T12:34:34.908389299Z Writing output script to '/opt/startup/startup.sh'
2025-06-19T12:34:34.985582864Z Using packages from virtual environment antenv located at /tmp/8ddaf2d5f5ba34f/antenv.
2025-06-19T12:34:34.985606418Z Updated PYTHONPATH to '/opt/startup/app_logs:/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages'
2025-06-19T12:34:34.989522610Z Starting Ooredoo Upsell Prediction App...
2025-06-19T12:34:40.775536899Z Traceback (most recent call last):
2025-06-19T12:34:40.775591991Z File "<string>", line 1, in <module>
2025-06-19T12:34:40.776540320Z File "/home/site/wwwroot/app.py", line 512, in <module>
2025-06-19T12:34:40.779120771Z init_database()
2025-06-19T12:34:40.779135138Z File "/home/site/wwwroot/app.py", line 45, in init_database
2025-06-19T12:34:40.779513440Z cursor.execute('''
2025-06-19T12:34:40.779524360Z sqlite3.DatabaseError: database disk image is malformed
2025-06-19T12:34:40.985771887Z Starting Flask application on port 8000...
2025-06-19T12:34:41.179096709Z [2025-06-19 12:34:41 +0000] [1080] [INFO] Starting gunicorn 21.2.0
2025-06-19T12:34:41.179821224Z [2025-06-19 12:34:41 +0000] [1080] [INFO] Listening at: http://0.0.0.0:8000 (1080)
2025-06-19T12:34:41.179834479Z [2025-06-19 12:34:41 +0000] [1080] [INFO] Using worker: sync
2025-06-19T12:34:41.190259135Z [2025-06-19 12:34:41 +0000] [1081] [INFO] Booting worker with pid: 1081
2025-06-19T12:34:41.255479822Z [2025-06-19 12:34:41 +0000] [1082] [INFO] Booting worker with pid: 1082
2025-06-19T12:34:43.583678857Z [2025-06-19 12:34:43 +0000] [1081] [ERROR] Exception in worker process
2025-06-19T12:34:43.583738027Z Traceback (most recent call last):
2025-06-19T12:34:43.583744930Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/arbiter.py", line 609, in spawn_worker
2025-06-19T12:34:43.583750981Z worker.init_process()
2025-06-19T12:34:43.583756551Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/workers/base.py", line 134, in init_process
2025-06-19T12:34:43.583761861Z self.load_wsgi()
2025-06-19T12:34:43.583766961Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/workers/base.py", line 146, in load_wsgi
2025-06-19T12:34:43.583772421Z self.wsgi = self.app.wsgi()
2025-06-19T12:34:43.583777730Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/app/base.py", line 67, in wsgi
2025-06-19T12:34:43.583783151Z self.callable = self.load()
2025-06-19T12:34:43.583788060Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/app/wsgiapp.py", line 58, in load
2025-06-19T12:34:43.583793049Z return self.load_wsgiapp()
2025-06-19T12:34:43.583798038Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/app/wsgiapp.py", line 48, in load_wsgiapp
2025-06-19T12:34:43.583803188Z return util.import_app(self.app_uri)
2025-06-19T12:34:43.583808207Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/util.py", line 371, in import_app
2025-06-19T12:34:43.583814048Z mod = importlib.import_module(module)
2025-06-19T12:34:43.583819558Z File "/opt/python/3.9.22/lib/python3.9/importlib/**init**.py", line 127, in import_module
2025-06-19T12:34:43.583824998Z return \_bootstrap.\_gcd_import(name[level:], package, level)
2025-06-19T12:34:43.583830058Z File "<frozen importlib._bootstrap>", line 1030, in \_gcd_import
2025-06-19T12:34:43.583848252Z File "<frozen importlib._bootstrap>", line 1007, in \_find_and_load
2025-06-19T12:34:43.583853972Z File "<frozen importlib._bootstrap>", line 986, in \_find_and_load_unlocked
2025-06-19T12:34:43.583858992Z File "<frozen importlib._bootstrap>", line 680, in \_load_unlocked
2025-06-19T12:34:43.583864201Z File "<frozen importlib._bootstrap_external>", line 850, in exec_module
2025-06-19T12:34:43.583869541Z File "<frozen importlib._bootstrap>", line 228, in \_call_with_frames_removed
2025-06-19T12:34:43.583875121Z File "/home/site/wwwroot/app.py", line 512, in <module>
2025-06-19T12:34:43.583880601Z init_database()
2025-06-19T12:34:43.583885421Z File "/home/site/wwwroot/app.py", line 45, in init_database
2025-06-19T12:34:43.583890750Z cursor.execute('''
2025-06-19T12:34:43.583895920Z sqlite3.DatabaseError: database disk image is malformed
2025-06-19T12:34:43.583901210Z [2025-06-19 12:34:43 +0000] [1081] [INFO] Worker exiting (pid: 1081)
2025-06-19T12:34:43.598871335Z [2025-06-19 12:34:43 +0000] [1082] [ERROR] Exception in worker process
2025-06-19T12:34:43.598902713Z Traceback (most recent call last):
2025-06-19T12:34:43.598909095Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/arbiter.py", line 609, in spawn_worker
2025-06-19T12:34:43.598914706Z worker.init_process()
2025-06-19T12:34:43.598920116Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/workers/base.py", line 134, in init_process
2025-06-19T12:34:43.598925596Z self.load_wsgi()
2025-06-19T12:34:43.598930866Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/workers/base.py", line 146, in load_wsgi
2025-06-19T12:34:43.598936627Z self.wsgi = self.app.wsgi()
2025-06-19T12:34:43.598942387Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/app/base.py", line 67, in wsgi
2025-06-19T12:34:43.598947316Z self.callable = self.load()
2025-06-19T12:34:43.598951915Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/app/wsgiapp.py", line 58, in load
2025-06-19T12:34:43.598957666Z return self.load_wsgiapp()
2025-06-19T12:34:43.598963036Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/app/wsgiapp.py", line 48, in load_wsgiapp
2025-06-19T12:34:43.598968746Z return util.import_app(self.app_uri)
2025-06-19T12:34:43.598973986Z File "/tmp/8ddaf2d5f5ba34f/antenv/lib/python3.9/site-packages/gunicorn/util.py", line 371, in import_app
2025-06-19T12:34:43.599142118Z mod = importlib.import_module(module)
2025-06-19T12:34:43.599151515Z File "/opt/python/3.9.22/lib/python3.9/importlib/**init\_\_.py", line 127, in import_module
2025-06-19T12:34:43.599156785Z return \_bootstrap.\_gcd_import(name[level:], package, level)
2025-06-19T12:34:43.599161474Z File "<frozen importlib._bootstrap>", line 1030, in \_gcd_import
2025-06-19T12:34:43.599180810Z File "<frozen importlib._bootstrap>", line 1007, in \_find_and_load
2025-06-19T12:34:43.599187242Z File "<frozen importlib._bootstrap>", line 986, in \_find_and_load_unlocked
2025-06-19T12:34:43.599192942Z File "<frozen importlib._bootstrap>", line 680, in \_load_unlocked
2025-06-19T12:34:43.599199084Z File "<frozen importlib._bootstrap_external>", line 850, in exec_module
2025-06-19T12:34:43.599227757Z File "<frozen importlib._bootstrap>", line 228, in \_call_with_frames_removed
2025-06-19T12:34:43.599233467Z File "/home/site/wwwroot/app.py", line 512, in <module>
2025-06-19T12:34:43.599238747Z init_database()
2025-06-19T12:34:43.599243717Z File "/home/site/wwwroot/app.py", line 45, in init_database
2025-06-19T12:34:43.599248906Z cursor.execute('''
2025-06-19T12:34:43.599253885Z sqlite3.DatabaseError: database disk image is malformed
2025-06-19T12:34:43.599262051Z [2025-06-19 12:34:43 +0000] [1082] [INFO] Worker exiting (pid: 1082)
2025-06-19T12:34:43.919233427Z [2025-06-19 12:34:43 +0000] [1080] [ERROR] Worker (pid:1082) exited with code 3
2025-06-19T12:34:43.919934078Z [2025-06-19 12:34:43 +0000] [1080] [ERROR] Worker (pid:1081) exited with code 3
2025-06-19T12:34:43.920825463Z [2025-06-19 12:34:43 +0000] [1080] [ERROR] Shutting down: Master
2025-06-19T12:34:43.921137864Z [2025-06-19 12:34:43 +0000] [1080] [ERROR] Reason: Worker failed to boot.
