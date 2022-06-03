var objShell = new ActiveXObject("Shell.Application");
        objShell.ShellExecute("cmd.exe", "C: cd C:\\pr main.exe blablafile.txt auto", "C:\\WINDOWS\\system32", "open", "1");