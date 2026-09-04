namespace nGroup.Info.eEvolution.FileSystem.FileSystemUi.ViewModels
{
  using System;
  using System.Collections.Generic;
  using System.ComponentModel;
  using System.IO;
  using System.Linq;
  using System.Windows.Forms;
  using nGroup.Info.eEvolution.Tools.Collections;
  using nGroup.Info.eEvolution.Tools.Extensions;
  using nGroup.Info.eEvolution.Windows.Infrastructure.Interface.Services.Ui;
  using nGroup.Info.eEvolution.Windows.Infrastructure.ViewModels;
  using Wisej.Core;

  public class FileSystemTreeViewModel : ViewModelDictionary, IParameterizedUpdatableViewModel<FileSystemTreeViewModel>
  {
    #region Fields

    private const string IMAGEKEY_DRIVE = "_drive_";
    private const string IMAGEKEY_FOLDER = "_folder_";
    private const string NOTLOADED_PLACEHOLDER = "$_NOT_LOADED_$";
    private ICursorManager cursorManager;
    private NodePathRegistry pathRegistry;

    /// <summary>
    /// Wird ausgelöst wenn der gewählte Pfad sich geändert hat.
    /// </summary>
    private Action<string> pathSelectedCallback;

    private IFileSystemProvider[] provider;

    #endregion Fields

    #region Constructors

    public FileSystemTreeViewModel()
    {
      this.DirectoryList = new BindingList<DirectoryItem>();
      this.pathRegistry = new NodePathRegistry();
    }

    #endregion Constructors

    #region Properties

    /// <summary>
    /// Die Liste der Bilder in der Ordnerauswahl.
    /// </summary>
    public ImageList FolderImageList
    {
      get
      {
        return this.GetValue<ImageList>();
      }

      set
      {
        this.SetValue(value);
      }
    }

    /// <summary>
    /// Die dargestellten Ordner.
    /// </summary>
    public BindingList<DirectoryItem> DirectoryList
    {
      get
      {
        return this.GetValue<BindingList<DirectoryItem>>();
      }

      set
      {
        this.SetValue(value);
      }
    }

    /// <summary>
    /// Der ausgewähte Ordner.
    /// </summary>
    public DirectoryItem SelectedDirectory
    {
      get
      {
        return this.GetValue<DirectoryItem>();
      }

      set
      {
        this.SetValue(value);
      }
    }


    #endregion Properties

    #region Methods

    /// <summary>
    /// Initialisiert das View Model.
    /// </summary>
    /// <param name="p">Die Initialisierungsparameter.</param>
    public void Initialize(IViewModelParameters<FileSystemTreeViewModel> p)
    {
      this.cursorManager = this.Resolve<ICursorManager>();

      var parameters = p as Parameters;
      if (parameters != null)
      {
        // Nichts zu tun
      }
    }

    public List<DirectoryItem> LoadDirElement(int nodeId)
    {
      var children = this.GetSubDirectorys(nodeId);
      if (children.Count != 1)
      {
        return children;
      }

      var singleChild = children.First();
      if (singleChild.Name != NOTLOADED_PLACEHOLDER)
      {
        return children;
      }

      var path = this.pathRegistry.GetPath(nodeId);

      var fsp = this.GetContainingFsp(path);

      var subDirs = fsp.GetDirectories(path, "*");

      var directChildren = subDirs.Select(name => new DirectoryItem
      {
        Name = Path.GetFileName(name),
        ParentDirId = nodeId,
        ImageIndex = this.GetImageIndex(IMAGEKEY_FOLDER)
      }).ToList();
      var newChildren = directChildren.SelectMany(this.SetDirIdAndCreateDummyItem).ToList();

      this.DirectoryList.Remove(singleChild);
      this.DirectoryList.AddRange(newChildren);

      return directChildren;
    }

    /// <summary>
    /// Aktualisiert das View Model.
    /// </summary>
    /// <param name="p">Die Aktualisierungsparameter.</param>
    public void Update(IViewModelParameters<FileSystemTreeViewModel> p)
    {
      var parameters = p as Parameters;

      if (parameters != null)
      {
        this.FolderImageList = this.CreateImageList(parameters.Provider);

        this.cursorManager.SetWaitCursor(true);

        this.DirectoryList.AddRange(this.CreateFromFsp(parameters.Provider));
        this.provider = parameters.Provider;

        this.cursorManager.SetWaitCursor(false);

        this.PropertyChanged -= this.SpecifyFileViewModel_PropertyChanged;
        this.PropertyChanged += this.SpecifyFileViewModel_PropertyChanged;

        this.pathSelectedCallback = parameters.PathSelectedCallback;

        if (!string.IsNullOrEmpty(parameters.InitialPath))
          this.Navigate(parameters.InitialPath);
      }
    }

    private DirectoryItem CreateDummyItem(DirectoryItem item)
    {
      return new DirectoryItem { Name = NOTLOADED_PLACEHOLDER, ParentDirId = item.DirId };
    }

    private List<DirectoryItem> CreateFromFsp(IFileSystemProvider[] provider)
    {
      var list = provider.Select(p => new DirectoryItem()
      {
        Name = p.Name,
        ImageIndex = this.GetImageIndex(p.Icon != null ? p.Name : IMAGEKEY_DRIVE),
        ParentDirId = 0
      }).SelectMany(this.SetDirIdAndCreateDummyItem).ToList();

      return list;
    }

    private ImageList CreateImageList(IFileSystemProvider[] provider)
    {
      ImageList list = new ImageList();

      list.Images.Add(IMAGEKEY_DRIVE, Base.IconRetriever.GetFileIcon("C:"));
      list.Images.Add(IMAGEKEY_FOLDER, Base.IconRetriever.GetFolderIcon(Base.IconRetriever.IconSize.Large, Base.IconRetriever.FolderType.Open));
      //list.Images.Add(Base.IconRetriever.GetFileIcon(".bak", true));

      foreach (var item in provider)
      {
        if (item.Icon == null)
        {
          continue;
        }

        list.Images.Add(item.Name, item.Icon);
      }

      return list;
    }

    private IFileSystemProvider GetContainingFsp(string path)
    {
      return this.provider.First(p => p.Contains(path));
    }

    private int GetImageIndex(string key)
    {
      return this.FolderImageList.Images.IndexOfKey(key);
    }

    private List<DirectoryItem> GetSubDirectorys(int nodeId)
    {
      return this.DirectoryList.Where(item => item.ParentDirId == nodeId).ToList(); // TODO: Etwas performanteres?
    }

    private void Navigate(string initialPath)
    {
      var parts = initialPath.Split(Path.DirectorySeparatorChar);
      DirectoryItem directoryItem = null;
      var children = this.DirectoryList.Where(f => f.ParentDirId == 0);
      foreach (var folderName in parts.Skip(1))
      {
        directoryItem = children.FirstOrDefault(c => c.Name == folderName);
        if (directoryItem == null)
          break;

        directoryItem.IsExpanded = true;
        children = this.LoadDirElement(directoryItem.DirId);
      }

      this.SelectedDirectory = directoryItem ?? this.SelectedDirectory;
    }

    private DirectoryItem SetDirId(DirectoryItem item)
    {
      item.DirId = this.pathRegistry.GetOrCreateNodeId(item.ParentDirId, item.Name);
      return item;
    }

    private IEnumerable<DirectoryItem> SetDirIdAndCreateDummyItem(DirectoryItem item)
    {
      yield return this.SetDirId(item);
      yield return this.SetDirId(this.CreateDummyItem(item));
    }

    private void SpecifyFileViewModel_PropertyChanged(object sender, PropertyChangedEventArgs e)
    {
      if (e.PropertyName == nameof(this.SelectedDirectory))
      {
        var path = this.pathRegistry.GetPath(this.SelectedDirectory.DirId);
        this.pathSelectedCallback?.Invoke(path);
      }
    }

    #endregion Methods

    #region Classes

    public sealed class NodePathRegistry
    {
      #region Fields

      private const string DIRECTORY_SEPERATOR = "/";
      private readonly BiDictionary<string, int> nodes = new();
      private readonly object syncRoot = new();
      private int nextNodeId = 1;

      #endregion Fields

      #region Methods

      public int GetOrCreateNodeId(int parentNodeId, string folderName)
      {
        if (folderName.Contains(DIRECTORY_SEPERATOR))
        {
          throw new Exception("Ordnername enthält illegales Zeichen.");
        }

        if (parentNodeId == 0)
        {
          return this.GetOrCreateNodeId(folderName);
        }

        string fullPath = this.GetPath(parentNodeId) + DIRECTORY_SEPERATOR + folderName;

        return this.GetOrCreateNodeId(fullPath);
      }

      public int GetOrCreateNodeId(string path)
      {
        lock (this.syncRoot)
        {
          if (this.nodes.TryGetValue(path, out int nodeId))
          {
            return nodeId;
          }

          nodeId = this.nextNodeId++;

          this.nodes.Add(path, nodeId);

          return nodeId;
        }
      }

      public int GetNodeId(string path) => this.nodes[path];

      public string GetPath(int nodeId) => this.nodes.Reversed[nodeId];

      #endregion Methods
    }

    /// <summary>
    /// Die Parameter für das Initialisieren und Aktualisieren.
    /// </summary>
    public class Parameters : IViewModelParameters<FileSystemTreeViewModel>
    {
      #region Properties

      public string InitialPath { get; internal set; }
      public Action<string> PathSelectedCallback { get; set; }
      public IFileSystemProvider[] Provider { get; set; }

      #endregion Properties
    }

    #endregion Classes
  }
}