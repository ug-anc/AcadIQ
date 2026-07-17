export default function Header() {
  return (
    <header className="w-full border-b border-slate-200 bg-white/90 backdrop-blur-sm mb-2">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <img
            src="https://anc-website-blond.vercel.app/_next/image?url=%2Fimages%2Flogo%2Fanc_logo2.png&w=128&q=75"
            alt="ANC Council logo"
            className="h-10 w-10 "
          />
          <div className="flex flex-col">
            <span className="text-lg font-semibold tracking-tight text-slate-900">
              Ask IITK
            </span>
            <span className="text-sm text-slate-500">ANC Council</span>
          </div>
        </div>
      </div>
    </header>
  );
}
