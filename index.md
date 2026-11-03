---
layout: default
---

<header class="site-header container">
  <a href="#top" class="brand">
    <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><defs><clipPath id="mk"><circle cx="13" cy="13" r="11.5"></circle></clipPath></defs><circle cx="13" cy="13" r="11.5" fill="none" stroke="currentColor" stroke-width="1.2"></circle><g clip-path="url(#mk)" stroke="currentColor" stroke-width="1.1"><line x1="0" y1="14" x2="26" y2="14"></line><line x1="0" y1="16.5" x2="26" y2="16.5"></line><line x1="0" y1="19" x2="26" y2="19"></line><line x1="0" y1="21.5" x2="26" y2="21.5"></line><line x1="0" y1="24" x2="26" y2="24"></line></g><circle cx="17" cy="9" r="3" fill="currentColor"></circle></svg>
    <span>Luthfi Raditya Meza</span>
  </a>
  <nav class="site-nav" aria-label="Main">
    <a href="#about">About</a>
    <a href="#projects">Projects</a>
    <a href="#experience">Experience</a>
    <a href="#contact">Contact</a>
  </nav>
</header>

<section id="top" class="hero container">
  <p class="mono hero-eyebrow">Senior Data Engineer · Jakarta</p>
  <h1 class="hero-title">I build the pipelines that carry a company's data, from source system to boardroom.</h1>
  <p class="hero-lede">4 years building production lakehouse platforms on AWS and Databricks for enterprises in retail, healthcare, telco and automotive.</p>
  <div class="hero-actions">
    <a href="#projects" class="tap">See selected work →</a>
    <a href="#contact" class="tap">Get in touch</a>
  </div>
</section>

<figure class="plate">
  <img src="{{ '/assets/img/plate-balloon.png' | relative_url }}" alt="Engraved illustration of a striped hot air balloon rising above an open field, tethered by a rope, with small figures in 19th century dress watching">
  <figcaption class="container">
    <span>Plate I. The ascent: careful groundwork, then lift.</span>
    <span>Fig. 01</span>
  </figcaption>
</figure>

<section id="about" class="section section-split container">
  <div class="section-label">
    <p class="mono">01</p>
    <h2>About</h2>
  </div>
  <div class="section-body">
    <p class="about-lead">Senior data engineer at Insignia, leading delivery across concurrent enterprise engagements and owning Databricks architecture standards company-wide.</p>
    <p class="about-text">My core is AWS lakehouse architecture (Glue/Spark, Iceberg, Lake Formation), Databricks (Unity Catalog, Delta Sharing, Lakeflow), Airflow and CDC pipelines. Before that I worked closer to the ML side: LLM reporting, NER models and vector search, which still shapes how I design data for AI workloads.</p>
    <div class="skill-columns">
      <div class="skill-column">
        <h3>Cloud and platform</h3>
        <span>AWS: Glue/Spark, S3, Athena</span><span>Lake Formation, Lambda, DMS</span><span>Databricks, Snowflake</span>
      </div>
      <div class="skill-column">
        <h3>Processing</h3>
        <span>Spark, Airflow, Iceberg</span><span>Kafka, Kinesis</span><span>CDC (DMS), Airbyte</span>
      </div>
      <div class="skill-column">
        <h3>Data and ops</h3>
        <span>Python, SQL</span><span>PostgreSQL, MongoDB, Redis</span><span>GitHub Actions, Docker, MLflow</span>
      </div>
    </div>
  </div>
</section>

<section id="projects" class="section container">
  <div class="section-head">
    <div>
      <p class="mono">02</p>
      <h2>Selected projects</h2>
    </div>
    <p class="note">Enterprise client work. Client names withheld; happy to walk through details in conversation.</p>
  </div>
  <div class="ruled-list">
    <article class="project">
      <p class="mono project-num">I.</p>
      <div class="project-title">
        <h3>Near real-time ingestion platform</h3>
        <p>Telco infrastructure</p>
      </div>
      <p class="project-desc">Architected an ingestion platform processing 10 to 15 GB a day from heterogeneous sources: SAP, RDBMS, POS, SFTP and flat files, landing into a governed lakehouse.</p>
      <p class="project-stack">AWS Glue · DMS · Iceberg · Airflow</p>
    </article>
    <article class="project">
      <p class="mono project-num">II.</p>
      <div class="project-title">
        <h3>7 TB historical migration</h3>
        <p>Automotive conglomerate</p>
      </div>
      <p class="project-desc">Led the migration of 7 TB of historical data: migration strategy, a validation framework to prove parity, and the cutover plan.</p>
      <p class="project-stack">Migration · Validation · Cutover</p>
    </article>
    <article class="project">
      <p class="mono project-num">III.</p>
      <div class="project-title">
        <h3>Data modernization program</h3>
        <p>Enterprise client</p>
      </div>
      <p class="project-desc">43 production pipelines and 30 GB monthly throughput, rebuilt on standardized ingestion patterns and medallion-layer conventions that were then reused across other projects.</p>
      <p class="project-stack">Databricks · Unity Catalog · Lakeflow</p>
    </article>
    <article class="project">
      <p class="mono project-num">IV.</p>
      <div class="project-title">
        <h3>Lakehouse lift-and-shift</h3>
        <p>Healthcare</p>
      </div>
      <p class="project-desc">Built and delivered a 30-pipeline migration into a lakehouse, covering data standardization, metadata centralization and schema governance.</p>
      <p class="project-stack">AWS · Lake Formation · Iceberg</p>
    </article>
    <article class="project">
      <p class="mono project-num">V.</p>
      <div class="project-title">
        <h3>CDC and streaming patterns</h3>
        <p>Unified Data Platform standard</p>
      </div>
      <p class="project-desc">Reusable CDC and streaming ingestion from Oracle, SQL Server and SFTP into an Iceberg lakehouse on AWS, shipped through GitHub Actions CI/CD and adopted as the team's default pattern.</p>
      <p class="project-stack">Glue · DMS · Kinesis · Athena</p>
    </article>
    <article class="project">
      <p class="mono project-num">VI.</p>
      <div class="project-title">
        <h3>LLM automated reporting</h3>
        <p>NTX Solusi Teknologi</p>
      </div>
      <p class="project-desc">Production LLM solution that cut client report creation time by 90%, backed by NER models at 95% accuracy and an HNSW vector search for retrieval.</p>
      <p class="project-stack">LangChain · MLflow · FastAPI</p>
    </article>
  </div>
</section>

<section id="experience" class="section section-split container">
  <div class="section-label">
    <p class="mono">03</p>
    <h2>Experience</h2>
  </div>
  <div class="section-body ruled-list">
    <div class="role">
      <p class="mono role-date">Jun 2025 to now</p>
      <div class="role-body">
        <h3>Senior Data Engineer · Insignia, Jakarta</h3>
        <p>Promoted after 3 months. Leads delivery across 4 concurrent enterprise engagements with a team of 7. Internal Databricks authority: reviewed and approved 4 project architectures. Set the team's engineering SOPs and led pre-sales that helped win two major enterprise accounts.</p>
      </div>
    </div>
    <div class="role">
      <p class="mono role-date">Apr to Jun 2025</p>
      <div class="role-body">
        <h3>Mid. Data Engineer · Insignia, Jakarta</h3>
        <p>Owned end-to-end delivery of platform modernization work. Defined architecture standards for client Unified Data Platforms, built AWS lakehouses on Iceberg, and introduced CI/CD for pipeline deployments.</p>
      </div>
    </div>
    <div class="role">
      <p class="mono role-date">Dec 2023 to Mar 2025</p>
      <div class="role-body">
        <h3>Data Engineer · NTX Solusi Teknologi, Jakarta</h3>
        <p>ETL/CDC pipelines and a MinIO data lake orchestrated with Prefect. Great Expectations validation caught 95% of anomalies before ingestion. Shipped an LLM reporting product, ran monitoring at 98% uptime, and rolled out DataHub and Amundsen for discovery.</p>
      </div>
    </div>
    <div class="role">
      <p class="mono role-date">Mar to Jun 2023</p>
      <div class="role-body">
        <h3>AI Engineer · PERGA, Hamburg (remote)</h3>
        <p>NER and document classification based on LayoutLM, reaching 92% accuracy at the first checkpoint.</p>
      </div>
    </div>
    <div class="role">
      <p class="mono role-date">2019 to 2023</p>
      <div class="role-body">
        <h3>B.Sc. Information Systems · Universitas Airlangga</h3>
        <p>Thesis: session-based recommendation system using gated graph neural networks and attention for e-commerce. Top 5 AI Startup Final Project, Orbit Future Academy.</p>
      </div>
    </div>
    <div class="role">
      <p class="mono role-date">Certifications</p>
      <div class="role-body list">
        <span>AWS Certified Data Engineer, 2025</span>
        <span>Databricks Certified Data Engineer Associate</span>
      </div>
    </div>
  </div>
</section>

<section id="contact" class="container">
  <p class="mono">04 · Contact</p>
  <h2>Have data that needs somewhere to go? Let's talk.</h2>
  <a href="mailto:luthfirdty@gmail.com" class="contact-email">luthfirdty@gmail.com</a>
  <div class="contact-links">
    <a href="https://www.linkedin.com/in/luthfiraditya/" class="tap">LinkedIn</a>
    <a href="https://medium.com/@luthfirdty" class="tap">Medium</a>
  </div>
</section>

<footer class="site-footer">
  <div class="footer-inner container">
    <div class="footer-about">
      <p class="footer-name">Luthfi Raditya Meza</p>
      <p class="footer-blurb">Senior data engineer building lakehouse platforms on AWS and Databricks. Based in Jakarta.</p>
      <p class="footer-copy">© {{ site.time | date: '%Y' }} Luthfi Raditya Meza</p>
    </div>
    <div class="footer-links">
      <nav aria-label="Sections">
        <h2>Index</h2>
        <a href="#about">About</a>
        <a href="#projects">Projects</a>
        <a href="#experience">Experience</a>
        <a href="#contact">Contact</a>
      </nav>
      <nav aria-label="Elsewhere">
        <h2>Elsewhere</h2>
        <a href="https://www.linkedin.com/in/luthfiraditya/">LinkedIn</a>
        <a href="https://medium.com/@luthfirdty">Medium</a>
        <a href="mailto:luthfirdty@gmail.com">Email</a>
      </nav>
    </div>
  </div>
  <img class="footer-plate" src="{{ '/assets/img/plate-pier.png' | relative_url }}" alt="Engraved illustration of a long wooden pier reaching into a calm sea, people strolling on it, a sailboat nearby">
</footer>
