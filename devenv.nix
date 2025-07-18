{ pkgs, lib, ... }:

{
  packages = [ 
    pkgs.just 
  ];

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
    venv.enable = true;
  };
}
