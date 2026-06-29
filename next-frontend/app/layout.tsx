import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ask IITK — Academic Policy Assistant",
  description: "Answers only from official college documents — every claim cited.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" style={{ height: "100%" }}>
      <body style={{ minHeight: "100%", display: "flex", flexDirection: "column" }}>
        {children}
      </body>
    </html>
  );
}
