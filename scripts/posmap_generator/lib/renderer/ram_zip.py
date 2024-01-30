import zipfile
import io
import os


class ZipPool(object):
    def __init__(self, append=False):
        self.files = dict()
        self._append_on_disk = append

    def write_str(self, prefix, path, data):
        zip_file = prefix + '.zip'
        post_fix = os.path.relpath(path, prefix)
        if zip_file not in self.files:
            if self._append_on_disk:
                self.files[zip_file] = OnDiskZip(zip_file)
            else:
                self.files[zip_file] = OnDiskWZip(zip_file)
        zp = self.files[zip_file]
        zp.append_str(post_fix, data)

    def write(self, prefix, path, file_path):
        zip_file = prefix + '.zip'
        post_fix = os.path.relpath(path, prefix)
        if zip_file not in self.files:
            if self._append_on_disk:
                self.files[zip_file] = OnDiskZip(zip_file)
            else:
                self.files[zip_file] = OnDiskWZip(zip_file)
        zp = self.files[zip_file]
        zp.append(post_fix, file_path)

    def flush(self):
        for zip_file in self.files:
            self.files[zip_file].writetofile()


class UnzipPool(object):
    def __init__(self, append=False):
        self.files = dict()
        self._append_on_disk = append

    def open(self, prefix, path):
        zip_file = prefix + '.zip'
        post_fix = os.path.relpath(path, prefix)
        post_fix = post_fix.replace('\\', '/')
        if zip_file not in self.files:
            self.files[zip_file] = UnZip(zip_file)
        zp = self.files[zip_file]
        return zp.open(post_fix)

    def read(self, prefix, path):
        zip_file = prefix + '.zip'
        post_fix = os.path.relpath(path, prefix)
        post_fix = post_fix.replace('\\', '/')
        if zip_file not in self.files:
            self.files[zip_file] = UnZip(zip_file)
        zp = self.files[zip_file]
        return zp.read(post_fix)

    def listdir(self, prefix, path):
        zip_file = prefix + '.zip'
        post_fix = os.path.relpath(path, prefix)
        post_fix = post_fix.replace('\\', '/')
        if zip_file not in self.files:
            self.files[zip_file] = UnZip(zip_file)
        zp = self.files[zip_file]
        file_names = zp.zf.namelist()

        result = set()
        for file_name in file_names:
            f = os.path.relpath(file_name, post_fix).replace('\\', '/').split('/')[0]
            if not f.startswith('.'):
                result.add(f)
        return result

    def close(self):
        for zip in self.files:
            self.files[zip].close()


class UnZip(object):
    def __init__(self, filename):
        # Create the in-memory file-like object
        self.zf = zipfile.ZipFile(filename, "r")

    def open(self, filename_in_zip):
        return self.zf.open(filename_in_zip)

    def read(self, filename_in_zip):
        return self.zf.read(filename_in_zip)

    def close(self):
        return self.zf.close()

class InMemoryZip(object):
    def __init__(self, filename):
        # Create the in-memory file-like object
        self.in_memory_zip = io.BytesIO()
        self.filename = filename
        self.zf = zipfile.ZipFile(self.in_memory_zip, "w", zipfile.ZIP_DEFLATED, True)

    def append_str(self, filename_in_zip, file_contents):
        '''Appends a file with name filename_in_zip and contents of
        file_contents to the in-memory zip.'''
        # Get a handle to the in-memory zip in append mode
        zf = self.zf

        # Write the file to the in-memory zip
        zf.writestr(filename_in_zip, file_contents)

        # Mark the files as having been created on Windows so that
        # Unix permissions are not inferred as 0000
        for zfile in zf.filelist:
            zfile.create_system = 0

        return self

    def read(self):
        '''Returns a string with the contents of the in-memory zip.'''
        self.in_memory_zip.seek(0)
        return self.in_memory_zip.read()

    def writetofile(self):
        '''Writes the in-memory zip to a file.'''
        self.zf.close()
        f = open(self.filename, "wb")
        f.write(self.read())
        f.close()


class OnDiskWZip(object):
    def __init__(self, filename):
        # Create the in-memory file-like object
        self.zf = zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED, True)

    def append_str(self, filename_in_zip, file_contents):
        '''Appends a file with name filename_in_zip and contents of
        file_contents to the in-memory zip.'''
        # Get a handle to the in-memory zip in append mode
        zf = self.zf

        # Write the file to the in-memory zip
        zf.writestr(filename_in_zip, file_contents)

        # Mark the files as having been created on Windows so that
        # Unix permissions are not inferred as 0000
        for zfile in zf.filelist:
            zfile.create_system = 0

        return self

    def append(self, filename_in_zip, file_path):
        zf = self.zf
        zf.write(file_path, filename_in_zip)
        for zfile in zf.filelist:
            zfile.create_system = 0

        return self


    def writetofile(self):
        '''Writes the in-memory zip to a file.'''
        self.zf.close()


class OnDiskZip(object):
    def __init__(self, filename):
        # Create the in-memory file-like object
        self.filename = filename

    def append(self, filename_in_zip, file_contents):
        '''Appends a file with name filename_in_zip and contents of
        file_contents to the in-memory zip.'''
        # Get a handle to the in-memory zip in append mode
        zf = zipfile.ZipFile(self.filename, "a", zipfile.ZIP_DEFLATED, True)

        # Write the file to the in-memory zip
        zf.writestr(filename_in_zip, file_contents)

        # Mark the files as having been created on Windows so that
        # Unix permissions are not inferred as 0000
        for zfile in zf.filelist:
            zfile.create_system = 0

        return self

    def writetofile(self):
        None
