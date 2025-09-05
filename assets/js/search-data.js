// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-jaxformer",
    title: "Jaxformer",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "dropdown-part-0-introduction",
              title: "Part 0. Introduction",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/";
              },
            },{id: "dropdown-part-1-tokenization",
              title: "Part 1. Tokenization",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/tokenization/";
              },
            },{id: "dropdown-part-2-base-model",
              title: "Part 2. Base Model",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/base_model/";
              },
            },{id: "dropdown-part-3-sharded-model",
              title: "Part 3. Sharded Model",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/sharded/";
              },
            },{id: "dropdown-part-4-dataset-amp-config",
              title: "Part 4. Dataset &amp; Config",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/dataset/";
              },
            },{id: "dropdown-part-5-distributed-training",
              title: "Part 5. Distributed Training",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/distributed_training/";
              },
            },{id: "dropdown-part-6-mixture-of-experts",
              title: "Part 6. Mixture of Experts",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/moe/";
              },
            },{id: "dropdown-part-7-training-results",
              title: "Part 7. Training Results",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/training/";
              },
            },{id: "dropdown-part-8-conclusion",
              title: "Part 8. Conclusion",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/conclusion/";
              },
            },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
